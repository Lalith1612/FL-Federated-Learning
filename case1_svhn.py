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
    # Load ResNet18
    # SVHN is 32x32 color images, so standard ResNet is fine
    model = resnet18(weights='IMAGENET1K_V1') 
    num_ftrs = model.fc.in_features
    # SVHN has 10 Classes (Digits 0-9)
    model.fc = nn.Linear(num_ftrs, 10)
    return model.to(DEVICE)

# 3. DATA LOADING (SVHN)
# SVHN is highly reliable on HF. using "svhn" with "cropped_digits" is standard
fds = FederatedDataset(dataset="svhn", subset="cropped_digits", partitioners={"train": 1})

def load_data(cid: int):
    print(f"Loading SVHN data for client {cid}...")
    
    # SVHN Mean/Std
    stats = ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)), # SVHN native size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    full_train = fds.load_split("train")
    full_test = fds.load_split("test")

    # SPLIT LOGIC: 10 Classes. 5 Clients.
    # Client 0: 0, 1
    # Client 1: 2, 3
    # ...
    classes_per_client = 2
    start_class = cid * classes_per_client
    end_class = start_class + classes_per_client
        
    client_classes = range(start_class, end_class)
    
    # Filter for Disjoint Classes
    client_train_hf = full_train.filter(lambda x: x['label'] in client_classes)
    
    # Wrapper
    class SVHNDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.hf_dataset = hf_dataset
            self.transform = transform
        def __len__(self):
            return len(self.hf_dataset)
        def __getitem__(self, idx):
            item = self.hf_dataset[idx]
            image = item["image"]
            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            label = item["label"]
            if self.transform:
                image = self.transform(image)
            return image, label

    train_set = SVHNDatasetWrapper(client_train_hf, transform=train_transform)
    test_set = SVHNDatasetWrapper(full_test, transform=test_transform)

    trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
    testloader = DataLoader(test_set, batch_size=32)

    print(f"  Client {cid}: {len(train_set)} train samples (Classes {list(client_classes)}).")
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
    if sum(examples) == 0:
        return {"accuracy": 0}
    return {"accuracy": sum(accuracies) / sum(examples)}

# 7. MAIN EXECUTION (Case 1 - Baseline SVHN)
if __name__ == "__main__":
    
    NUM_ROUNDS = 50
    NUM_CLIENTS = 5

    print("--- Starting Case 1 (Baseline) SVHN Simulation ---")
    print("--- Setup: Standard FedAvg (No Head Transfer) ---")

    # 1. Initialize Strategy
    net = load_model()
    initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()])
    
    strategy = fl.server.strategy.FedAvg(
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
    
    current_parameters = initial_parameters 
    status_ok = Status(code=Code.OK, message="Success")

    for server_round in range(1, NUM_ROUNDS + 1):
        round_start_time = time.time()
        print(f"\n[ROUND {server_round}/{NUM_ROUNDS}]")

        # FIT: Select one client sequentially
        client_index_to_train = (server_round - 1) % NUM_CLIENTS
        client = clients[client_index_to_train]
        print(f"  Training client {client_index_to_train}...")
        
        ndarrays_to_send = parameters_to_ndarrays(current_parameters)
        
        # Train
        ndarrays_returned, num_examples, metrics = client.fit(ndarrays_to_send, config={})
        
        # Aggregate
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
        plt.title("Case 1: Baseline FL Performance (SVHN)", fontsize=16)
        plt.savefig("case_1_svhn_50_rounds.png")
        print("Plot saved to case_1_svhn_50_rounds.png")