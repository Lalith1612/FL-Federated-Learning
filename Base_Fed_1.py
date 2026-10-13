# ==============================================================================
# IMPORTS
# ==============================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.models import resnet18
from torchvision import transforms
import flwr as fl
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
    EvaluateRes,
    Parameters,
    Status,  # <-- THE FIX: Import Status
    Code,    # <-- THE FIX: Import Code
)
from collections import OrderedDict
import warnings
import matplotlib.pyplot as plt
import time # We'll use this to time the rounds

# Suppress warnings for a cleaner output.
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. DEVICE CONFIGURATION
# ==============================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. Training will be extremely slow on CPU.")

# ==============================================================================
# 2. MODEL DEFINITION (USING RESNET-18)
# ==============================================================================
def load_model():
    """Load a ResNet18 model and adapt it for our 10-class problem."""
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model.to(DEVICE)

# ==============================================================================
# 3. DATA LOADING & TRANSFORMATION (WITH AUGMENTATION)
# ==============================================================================
def load_data(cid: int):
    """Load a unique dataset for each client with data augmentation."""
    print(f"Loading data for client {cid}...")
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Normalization for testing
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data_root = "./data"

    if cid == 0:
        dataset = CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        train_size = len(dataset) // 2
        trainset, _ = random_split(dataset, [train_size, len(dataset) - train_size])
    elif cid == 1:
        trainset = MNIST(root=data_root, train=True, download=True, transform=train_transform)
    elif cid == 2:
        trainset = FashionMNIST(root=data_root, train=True, download=True, transform=train_transform)
    else:
        dataset = CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        train_size = len(dataset) // 2
        _, second_half = random_split(dataset, [train_size, len(dataset) - train_size])
        client_3_size = len(second_half) // 2
        if cid == 3:
            trainset, _ = random_split(second_half, [client_3_size, len(second_half) - client_3_size])
        else:
            _, trainset = random_split(second_half, [client_3_size, len(second_half) - client_3_size])

    testset = CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    print(f"Finished loading data for client {cid}.")
    return trainloader, testloader

# ==============================================================================
# 4. TRAINING & TESTING UTILITIES (WITH LR SCHEDULER)
# ==============================================================================
def train(net, trainloader, epochs):
    """Train the model on the client's local data with a learning rate scheduler."""
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
        # We print epoch logs only if epochs > 1 to avoid clutter
        if epochs > 1:
            print(f"    Epoch {epoch+1}/{epochs} finished.")

def test(net, testloader):
    """Evaluate the model on the test set."""
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

# ==============================================================================
# 5. FLOWER CLIENT IMPLEMENTATION
# ==============================================================================
# This is a NumPyClient, it expects and returns List[np.ndarray]
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
        train(self.net, self.trainloader, epochs=7) # 7 local epochs
        print(f"  Training finished.")
        # Return results as (List[np.ndarray], int, dict)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        # Return results as (float, int, dict)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

# ==============================================================================
# 6. METRICS AGGREGATION FUNCTION (Used by Strategy)
# ==============================================================================
def weighted_average(metrics):
    """Aggregate evaluation results from multiple clients."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# ==============================================================================
# 7. MAIN EXECUTION BLOCK (MANUAL SIMULATION LOOP)
# ==============================================================================
if __name__ == "__main__":
    
    NUM_ROUNDS = 50
    NUM_CLIENTS = 5

    print("--- Starting Manual Federated Learning Simulation ---")

    # 1. Initialize Server Strategy
    print("Initializing server strategy (FedAvgM)...")
    net = load_model()
    # Get initial parameters as a Flower Parameters object
    initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()])
    
    strategy = fl.server.strategy.FedAvgM(
        fraction_fit=0.2,  # Train 1 client per round (0.2 * 5 clients = 1)
        min_fit_clients=1, # Minimum 1 client
        fraction_evaluate=0.5, # Evaluate on 2 clients (0.5 * 5 = 2.5, rounded down)
        min_evaluate_clients=2,
        min_available_clients=NUM_CLIENTS, # We need all 5 available
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average, # Give the strategy our helper function
    )
    print("Strategy initialized.")

    # 2. Create all clients
    print(f"Initializing {NUM_CLIENTS} clients...")
    clients = []
    for i in range(NUM_CLIENTS):
        net_client = load_model()
        trainloader, testloader = load_data(cid=i)
        client = FlowerClient(net_client, trainloader, testloader)
        clients.append(client)
    print("All clients initialized.")

    # 3. Initialize history
    history = {"losses_distributed": [], "metrics_distributed": {"accuracy": []}}

    # 4. Run the main training loop
    print(f"--- Starting main training loop for {NUM_ROUNDS} rounds ---")
    
    # current_parameters is a Flower Parameters object
    current_parameters = initial_parameters 
    status_ok = Status(code=Code.OK, message="Success") # Create a re-usable OK status

    for server_round in range(1, NUM_ROUNDS + 1):
        round_start_time = time.time()
        print(f"\n[ROUND {server_round}/{NUM_ROUNDS}]")

        # --- FIT (TRAINING) PHASE ---
        print("Fit phase (training clients)...")
        client_index_to_train = (server_round - 1) % NUM_CLIENTS
        client = clients[client_index_to_train]
        
        print(f"  Training client {client_index_to_train}...")
        
        ndarrays_to_send = parameters_to_ndarrays(current_parameters)
        
        # Call client.fit(), which returns (List[np.ndarray], int, dict)
        ndarrays_returned, num_examples, metrics = client.fit(ndarrays_to_send, config={})
        
        parameters_returned = ndarrays_to_parameters(ndarrays_returned)
        
        # *** THE FIX: Create a proper FitRes object with status ***
        fit_res = FitRes(
            status=status_ok, # <-- THE FIX
            parameters=parameters_returned,
            num_examples=num_examples,
            metrics=metrics,
        )
        
        aggregate_fit_res = strategy.aggregate_fit(
            server_round=server_round,
            results=[(client, fit_res)],
            failures=[]
        )
        
        if aggregate_fit_res:
            current_parameters, _ = aggregate_fit_res
            print("  Fit phase aggregated.")
        else:
            print("  Fit phase aggregation failed.")

        # --- EVALUATE (TESTING) PHASE ---
        print("Evaluate phase (testing model)...")
        
        ndarrays_for_eval = parameters_to_ndarrays(current_parameters)
        
        results_for_strategy = []
        aggregated_loss_for_history = 0.0 # To store the loss
        
        for i in range(2):
            client_to_eval = clients[i]
            print(f"  Evaluating on client {i}...")
            # client.evaluate() returns (float, int, dict)
            loss, num_examples_eval, metrics_eval = client_to_eval.evaluate(ndarrays_for_eval, config={})
            
            if i == 0: # Just grab the first client's loss for history
                aggregated_loss_for_history = loss

            # *** THE FIX: Create a proper EvaluateRes object with status ***
            evaluate_res = EvaluateRes(
                status=status_ok, # <-- THE FIX
                loss=loss,
                num_examples=num_examples_eval,
                metrics=metrics_eval,
            )
            results_for_strategy.append((client_to_eval, evaluate_res))
        
        aggregate_eval_res = strategy.aggregate_evaluate(
            server_round=server_round,
            results=results_for_strategy,
            failures=[]
        )
        
        if aggregate_eval_res:
            aggregated_loss, aggregated_metrics = aggregate_eval_res
            # *** THE FIX: Corrected variable name from aggregated_Loss to aggregated_loss ***
            print(f"  Evaluate phase aggregated: Loss: {aggregated_loss:.4f}, Accuracy: {aggregated_metrics['accuracy']:.4f}")
            
            # Save history
            history["losses_distributed"].append((server_round, aggregated_loss))
            history["metrics_distributed"]["accuracy"].append((server_round, aggregated_metrics['accuracy']))
        else:
            print("  Evaluate phase failed.")

        round_end_time = time.time()
        print(f"Round {server_round} finished in {round_end_time - round_start_time:.2f} seconds.")

    print("\n--- Federated Learning Simulation finished. ---")

    # ==========================================================================
    # 5. VISUALIZATION
    # ==========================================================================
    if history["metrics_distributed"]["accuracy"]:
        print("Plotting results...")
        
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
        plt.title("Case 1: Baseline FL Performance (ResNet18, 50 Rounds, Manual Loop)", fontsize=16)
        plt.xticks(range(0, 51, 5))
        plt.legend(loc='best')
        
        # Save the plot in the current directory
        plt.savefig("case_1_baseline_50_rounds.png")
        print("Plot saved to case_1_baseline_50_rounds.png")
        
        print("Plot saved, not attempting to show interactively.")
    else:
        print("No distributed metrics found in history to plot.")
