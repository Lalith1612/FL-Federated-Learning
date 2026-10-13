import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18
from torchvision import transforms
import flwr as fl
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
    EvaluateRes,
    Parameters,
    Status,
    Code,
)
from flwr_datasets import FederatedDataset # <-- The New Library
from collections import OrderedDict
import warnings
import matplotlib.pyplot as plt
import time

warnings.filterwarnings("ignore", category=UserWarning)

# 1. DEVICE
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")

# 2. MODEL (ResNet-18 for 100 Classes)
def load_model():
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # CIFAR-100 has 100 classes
    model.fc = nn.Linear(num_ftrs, 100)
    return model.to(DEVICE)

# 3. DATA LOADING (Using flwr-datasets)
# We initialize the FederatedDataset helper.
# This handles downloading and caching automatically.
fds = FederatedDataset(dataset="cifar100", partitioners={"train": 1})

def load_data(cid: int):
    print(f"Loading CIFAR-100 data for client {cid}...")
    
    # 1. Define Transforms
    # We use the standard CIFAR mean/std normalization
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # 2. Load the Full Dataset using Flower Datasets
    # This returns a Hugging Face Dataset object
    full_train = fds.load_split("train")
    full_test = fds.load_split("test")

    # 3. Strict Non-IID Partitioning (Disjoint Classes)
    # Total 100 classes. 5 Clients. Each gets 20 classes.
    # Client 0: 0-19, Client 1: 20-39, ...
    start_class = cid * 20
    end_class = start_class + 20
    client_classes = range(start_class, end_class)
    
    print(f"  Client {cid} gets classes: {start_class} to {end_class-1}")

    # We filter the Hugging Face dataset to keep only this client's classes
    # 'fine_label' is the column name for the 100 classes in CIFAR-100
    # Note: This filtering happens in memory/disk via HF Datasets
    client_train_hf = full_train.filter(lambda x: x['fine_label'] in client_classes)
    
    # 4. Apply Transforms (Map function)
    def apply_train_transforms(batch):
        # HF datasets give images as PIL. Convert to Tensor.
        batch["img"] = [train_transform(img) for img in batch["img"]]
        return batch

    def apply_test_transforms(batch):
        batch["img"] = [test_transform(img) for img in batch["img"]]
        return batch

    client_train_hf = client_train_hf.with_transform(apply_train_transforms)
    # For testing, we generally test on the GLOBAL distribution (all classes) to check global performance
    full_test_hf = full_test.with_transform(apply_test_transforms)

    # 5. Create PyTorch DataLoaders
    # Note: We need a custom collate_fn or careful batching because HF returns dicts
    # PyTorch DataLoader works fine if the dataset returns tensors.
    
    # To bridge HF Dataset -> PyTorch DataLoader smoothly, we can wrapping it or iterate carefully.
    # A simple way is to use the `with_format("torch")` feature of HF datasets.
    
    # Let's keep it simple: we will iterate the HF dataset in the train loop manually or wrap it.
    # Creating a simple wrapper class is safest for PyTorch DataLoaders.
    
    class HFDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, hf_dataset):
            self.hf_dataset = hf_dataset
        def __len__(self):
            return len(self.hf_dataset)
        def __getitem__(self, idx):
            item = self.hf_dataset[idx]
            return item["img"], item["fine_label"]

    train_set = HFDatasetWrapper(client_train_hf)
    test_set = HFDatasetWrapper(full_test_hf)

    trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
    testloader = DataLoader(test_set, batch_size=32)

    print(f"  Client {cid}: {len(client_train_hf)} train samples.")
    return trainloader, testloader

# 4. TRAINING & TESTING UTILITIES
def train(net, trainloader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    net.train()
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        # if epochs > 1:
        #     print(f"    Epoch {epoch+1}/{epochs} finished.")

def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = loss / len(testloader)
    return avg_loss, accuracy

# 5. FLOWER CLIENT
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print(f"  Training for 7 epochs...")
        train(self.net, self.trainloader, epochs=7)
        print(f"  Training finished.")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# 7. MAIN EXECUTION (Case 1 - CIFAR-100 Baseline)
if __name__ == "__main__":
    
    NUM_ROUNDS = 50
    NUM_CLIENTS = 5

    print("--- Starting Case 1 (Baseline) CIFAR-100 Simulation ---")

    # 1. Initialize Strategy
    net = load_model()
    initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()])
    
    strategy = fl.server.strategy.FedAvgM(
        fraction_fit=0.2,
        min_fit_clients=1,
        fraction_evaluate=0.5,
        min_evaluate_clients=2,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # 2. Create Clients
    clients = []
    print("Initializing clients (this may take a moment to download/filter data)...")
    for i in range(NUM_CLIENTS):
        trainloader, testloader = load_data(cid=i)
        net_client = load_model()
        client = FlowerClient(net_client, trainloader, testloader)
        clients.append(client)
    
    # 3. Simulation Loop (Manual Loop for Safety)
    history = {"losses_distributed": [], "metrics_distributed": {"accuracy": []}}
    current_parameters = initial_parameters 
    status_ok = Status(code=Code.OK, message="Success")

    for server_round in range(1, NUM_ROUNDS + 1):
        round_start_time = time.time()
        print(f"\n[ROUND {server_round}/{NUM_ROUNDS}]")

        # FIT
        client_index_to_train = (server_round - 1) % NUM_CLIENTS
        client = clients[client_index_to_train]
        print(f"  Training client {client_index_to_train}...")
        
        ndarrays_to_send = parameters_to_ndarrays(current_parameters)
        ndarrays_returned, num_examples, metrics = client.fit(ndarrays_to_send, config={})
        parameters_returned = ndarrays_to_parameters(ndarrays_returned)
        
        fit_res = FitRes(status=status_ok, parameters=parameters_returned, num_examples=num_examples, metrics=metrics)
        aggregate_fit_res = strategy.aggregate_fit(server_round=server_round, results=[(client, fit_res)], failures=[])
        
        if aggregate_fit_res: current_parameters, _ = aggregate_fit_res

        # EVALUATE
        ndarrays_for_eval = parameters_to_ndarrays(current_parameters)
        results_for_strategy = []
        
        # Eval on 2 clients
        for i in range(2):
            client_to_eval = clients[i]
            # print(f"  Evaluating on client {i}...")
            loss, num_examples_eval, metrics_eval = client_to_eval.evaluate(ndarrays_for_eval, config={})
            evaluate_res = EvaluateRes(status=status_ok, loss=loss, num_examples=num_examples_eval, metrics=metrics_eval)
            results_for_strategy.append((client_to_eval, evaluate_res))
        
        aggregate_eval_res = strategy.aggregate_evaluate(server_round=server_round, results=results_for_strategy, failures=[])
        
        if aggregate_eval_res:
            aggregated_loss, aggregated_metrics = aggregate_eval_res
            print(f"  Loss: {aggregated_loss:.4f}, Accuracy: {aggregated_metrics['accuracy']:.4f}")
            history["losses_distributed"].append((server_round, aggregated_loss))
            history["metrics_distributed"]["accuracy"].append((server_round, aggregated_metrics['accuracy']))

        print(f"Round {server_round} finished in {time.time() - round_start_time:.2f} s.")

    # PLOTTING
    if history["metrics_distributed"]["accuracy"]:
        loss_distributed = history["losses_distributed"]
        accuracy_distributed = history["metrics_distributed"]["accuracy"]

        fig, ax1 = plt.subplots(figsize=(12, 7))
        rounds = [x[0] for x in loss_distributed]
        loss = [x[1] for x in loss_distributed]
        color = 'tab:red'
        ax1.set_xlabel('Federated Round', fontsize=14)
        ax1.set_ylabel('Centralized Loss', color=color, fontsize=14)
        ax1.plot(rounds, loss, color=color, marker='o', linestyle='dashed', label='Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax2 = ax1.twinx()
        acc_rounds = [x[0] for x in accuracy_distributed]
        accuracy = [x[1] * 100 for x in accuracy_distributed]
        color = 'tab:blue'
        ax2.set_ylabel('Centralized Accuracy (%)', color=color, fontsize=14)
        ax2.plot(acc_rounds, accuracy, color=color, marker='x', linestyle='solid', label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title("Case 1: Baseline FL Performance (CIFAR-100 Split)", fontsize=16)
        plt.savefig("case_1_cifar100_50_rounds.png")
        print("Plot saved to case_1_cifar100_50_rounds.png")