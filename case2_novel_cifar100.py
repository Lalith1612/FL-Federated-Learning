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
from flwr_datasets import FederatedDataset
import warnings
import matplotlib.pyplot as plt
import time
from collections import OrderedDict

warnings.filterwarnings("ignore", category=UserWarning)

# 1. DEVICE
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")

# 2. MODEL
def load_model():
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    return model.to(DEVICE)

# 3. DATA LOADING (Identical to Case 1)
fds = FederatedDataset(dataset="cifar100", partitioners={"train": 1})

def load_data(cid: int):
    print(f"Loading CIFAR-100 data for client {cid}...")
    
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

    full_train = fds.load_split("train")
    full_test = fds.load_split("test")

    start_class = cid * 20
    end_class = start_class + 20
    client_classes = range(start_class, end_class)
    
    # Filter for Disjoint Classes
    client_train_hf = full_train.filter(lambda x: x['fine_label'] in client_classes)
    
    def apply_train_transforms(batch):
        batch["img"] = [train_transform(img) for img in batch["img"]]
        return batch

    def apply_test_transforms(batch):
        batch["img"] = [test_transform(img) for img in batch["img"]]
        return batch

    client_train_hf = client_train_hf.with_transform(apply_train_transforms)
    full_test_hf = full_test.with_transform(apply_test_transforms)

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

# 4. TRAINING & TESTING
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

# 7. MAIN EXECUTION (Case 2 - Novel CIFAR-100)
if __name__ == "__main__":
    
    NUM_ROUNDS = 50
    NUM_CLIENTS = 5

    print("--- Starting Case 2 (Novel) CIFAR-100 Simulation ---")
    print("--- Setup: Communication Enabled (Head Transfer) ---")

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
    print("Initializing clients...")
    for i in range(NUM_CLIENTS):
        trainloader, testloader = load_data(cid=i)
        net_client = load_model()
        client = FlowerClient(net_client, trainloader, testloader)
        clients.append(client)

    history = {"losses_distributed": [], "metrics_distributed": {"accuracy": []}}
    
    # ==========================================================================
    # NOVELTY: The Communication Channel
    # ==========================================================================
    previous_client_head = None 

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
        
        # 2. COMMUNICATION: Inject Previous Head
        if previous_client_head is not None:
            print("  [COMMUNICATION] Injecting Head from previous client...")
            # ResNet18 fc layer: weights at [-2], bias at [-1]
            ndarrays_to_send[-2] = previous_client_head[0]
            ndarrays_to_send[-1] = previous_client_head[1]
        else:
            print("  [COMMUNICATION] No previous head available.")

        # 3. Train
        ndarrays_returned, num_examples, metrics = client.fit(ndarrays_to_send, config={})
        
        # 4. COMMUNICATION: Extract New Head
        print("  [COMMUNICATION] Extracting new Head for next client...")
        previous_client_head = (ndarrays_returned[-2], ndarrays_returned[-1])

        # 5. Aggregate
        parameters_returned = ndarrays_to_parameters(ndarrays_returned)
        fit_res = FitRes(status=status_ok, parameters=parameters_returned, num_examples=num_examples, metrics=metrics)
        aggregate_fit_res = strategy.aggregate_fit(server_round=server_round, results=[(client, fit_res)], failures=[])
        
        if aggregate_fit_res: current_parameters, _ = aggregate_fit_res

        # EVALUATE
        ndarrays_for_eval = parameters_to_ndarrays(current_parameters)
        results_for_strategy = []
        
        for i in range(2):
            client_to_eval = clients[i]
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
        plt.title("Case 2: Novel FL Performance (CIFAR-100 Head Transfer)", fontsize=16)
        plt.savefig("case_2_cifar100_50_rounds.png")
        print("Plot saved to case_2_cifar100_50_rounds.png")