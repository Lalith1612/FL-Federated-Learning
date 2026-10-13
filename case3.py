import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
from collections import OrderedDict
import warnings
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# 1. DEVICE
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")

# 2. MODEL (Simple MLP for Tabular Data)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=3, num_classes=3):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def load_model():
    # Input dim is 3 (Year, Mileage, HP), Classes is 3 (Cheap, Med, Exp)
    return SimpleMLP(input_dim=3, num_classes=3).to(DEVICE)

# 3. DATA LOADING (CSV)
def load_data(cid: int):
    print(f"Loading CSV data for client {cid}...")
    
    # Read the specific CSV for this client
    df = pd.read_csv(f"./data/client_{cid}.csv")
    
    # Features (X) and Target (y)
    # Assuming last column is target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize (Important for Neural Networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)
    
    print(f"Client {cid}: Loaded {len(train_dataset)} training samples.")
    return trainloader, testloader

# 4. TRAINING & TESTING
def train(net, trainloader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001) # Adam is better for CSVs
    net.train()
    for epoch in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(features), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = loss / len(testloader)
    return avg_loss, accuracy

# 5. CLIENT
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
        train(self.net, self.trainloader, epochs=5)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# 7. MAIN EXECUTION (Case 1 - CSV)
if __name__ == "__main__":
    
    NUM_ROUNDS = 50
    NUM_CLIENTS = 5

    print("--- Starting Case 1 (Baseline) CSV Simulation ---")

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
    for i in range(NUM_CLIENTS):
        net_client = load_model()
        trainloader, testloader = load_data(cid=i)
        client = FlowerClient(net_client, trainloader, testloader)
        clients.append(client)

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
        plt.title("Case 1: Baseline FL Performance (CSV Cars)", fontsize=16)
        plt.savefig("case_1_csv_cars_50_rounds.png")
        print("Plot saved to case_1_csv_cars_50_rounds.png")