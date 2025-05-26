from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import pandas as pd

def prepare_features(df):
    df_encoded = pd.get_dummies(df, columns=["Application_Type"], drop_first=True)

    features = df_encoded.drop(columns=[
    "Timestamp", "User_ID", "Resource_Allocation",
    "Anomalie_Latence", "Anomalie_Signal", "Anomalie_BW", "Anomalie_Global",
    "Type_Anomalie",  # ← ajoute ceci !
    "Latency_ms"
    ], errors="ignore")


    target = df["Latency_ms"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    y = target.values

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42), scaler

class DeepLatencyModel6Layers(nn.Module):
    def __init__(self, input_dim):
        super(DeepLatencyModel6Layers, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_latency_model(X_train, y_train, input_dim, epochs=1000, lr=0.001):
    model = DeepLatencyModel6Layers(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Époque {epoch+1}/{epochs} - Perte : {loss.item():.4f}")

    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
        true = y_test.numpy()

    # MSE
    mse = mean_squared_error(true, predictions)
    print(f"MSE du modèle classique : {mse:.4f}")
    

    os.makedirs("graphs", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.scatter(true, predictions, alpha=0.7)
    plt.xlabel("Latence réelle (ms)")
    plt.ylabel("Latence prédite (ms)")
    plt.title("Modèle profond (6 couches cachées)")
    plt.grid(True)
    plt.plot([min(true), max(true)], [min(true), max(true)], "r--")
    plt.tight_layout()
    plt.savefig("graphs/prediction_vs_reel.png")
    plt.close()

if __name__ == "__main__":
    from data_loading import load_and_prepare_data
    from anomalies import detect_anomalies

    df = load_and_prepare_data()
    df = detect_anomalies(df)
    (X_train, X_test, y_train, y_test), _ = prepare_features(df)
    model = train_latency_model(X_train, y_train, input_dim=X_train.shape[1])
    evaluate_model(model, X_test, y_test)
