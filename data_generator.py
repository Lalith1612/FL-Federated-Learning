import pandas as pd
import numpy as np
import os

# Create data directory
os.makedirs("./data", exist_ok=True)

def generate_client_data(cid):
    """Generates a synthetic car dataset with a bias based on client ID."""
    n_samples = 2000
    np.random.seed(cid)

    # Features: Year, Mileage, Horsepower
    years = np.random.randint(2000, 2024, n_samples)
    
    # Bias: Each client sees a different range of cars (Non-IID)
    if cid == 0: # Economy cars
        mileage = np.random.randint(50000, 150000, n_samples)
        horsepower = np.random.randint(70, 130, n_samples)
    elif cid == 1: # Luxury cars
        mileage = np.random.randint(10000, 60000, n_samples)
        horsepower = np.random.randint(200, 400, n_samples)
    elif cid == 2: # Older cars
        years = np.random.randint(1990, 2010, n_samples)
        mileage = np.random.randint(100000, 200000, n_samples)
        horsepower = np.random.randint(100, 200, n_samples)
    elif cid == 3: # Sport cars
        mileage = np.random.randint(5000, 40000, n_samples)
        horsepower = np.random.randint(300, 600, n_samples)
    else: # Average cars
        mileage = np.random.randint(30000, 100000, n_samples)
        horsepower = np.random.randint(120, 250, n_samples)

    # Target: Price Class (0: Cheap, 1: Medium, 2: Expensive)
    # Logic: Newer + Lower Mileage + Higher HP = Expensive
    score = (years - 1990) * 100 - (mileage * 0.01) + (horsepower * 10)
    
    # Create labels based on score quantiles
    labels = []
    thresholds = np.percentile(score, [33, 66])
    for s in score:
        if s < thresholds[0]: labels.append(0)
        elif s < thresholds[1]: labels.append(1)
        else: labels.append(2)

    df = pd.DataFrame({
        'year': years,
        'mileage': mileage,
        'horsepower': horsepower,
        'price_class': labels
    })
    
    filename = f"./data/client_{cid}.csv"
    df.to_csv(filename, index=False)
    print(f"Created {filename} with {len(df)} rows.")

print("Generating synthetic Car datasets...")
for i in range(5):
    generate_client_data(i)
print("Done. You can now run the FL simulation.")