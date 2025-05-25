from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.autograd import Function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os

def grad_reverse(x, lambda_=0.001):
    class GradientReversalFunction(Function):
        @staticmethod
        def forward(ctx, input):
            ctx.lambda_ = lambda_
            return input.view_as(input)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.lambda_
    return GradientReversalFunction.apply(x)

class GradientUnlearningModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GradientUnlearningModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.adversary = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, lambda_=0.001):
        shared_feat = self.shared(x)
        reverse_feat = grad_reverse(shared_feat, lambda_)
        return self.regressor(shared_feat), self.adversary(reverse_feat)

def prepare_data_for_unlearning(df):
    label_encoder = LabelEncoder()
    y_adv = label_encoder.fit_transform(df["Application_Type"])

    df_encoded = pd.get_dummies(df, columns=["Application_Type"], drop_first=True)

    features = df_encoded.drop(columns=[
        "Timestamp", "User_ID", "Latency_ms", "Resource_Allocation",
        "Anomalie_Latence", "Anomalie_Signal", "Anomalie_BW", "Anomalie_Global","Type_Anomalie"
    ], errors="ignore")
    target = df["Latency_ms"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    y = target.values

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    y_adv_tensor = torch.tensor(y_adv, dtype=torch.long)

    return train_test_split(X_tensor, y_tensor, y_adv_tensor, test_size=0.2, random_state=42), label_encoder

def train_unlearning_model(X_train, y_train, y_train_adv, input_dim, num_classes, epochs=600):
    model = GradientUnlearningModel(input_dim, num_classes)
    criterion_reg = nn.MSELoss()
    criterion_adv = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        pred_latency, pred_app_type = model(X_train, lambda_=0.001)

        loss_latency = criterion_reg(pred_latency, y_train)
        loss_adversarial = criterion_adv(pred_app_type, y_train_adv)

        total_loss = loss_latency - 0.1 * loss_adversarial

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Latency Loss: {loss_latency.item():.4f}, Adversarial Loss: {loss_adversarial.item():.4f}")

    return model

def evaluate_unlearning(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        pred_latency, _ = model(X_test)
        predictions = pred_latency.numpy()
        true = y_test.numpy()

    mse_unlearn = mean_squared_error(true, predictions)
    print(f"📉 MSE du modèle avec unlearning (GRL) : {mse_unlearn:.4f}")    

    os.makedirs("graphs", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.scatter(true, predictions, alpha=0.7)
    plt.xlabel("Latence réelle (ms)")
    plt.ylabel("Latence prédite (ms)")
    plt.title("Modèle avec Gradient Reversal (Unlearning)")
    plt.grid(True)
    plt.plot([min(true), max(true)], [min(true), max(true)], "r--")
    plt.tight_layout()
    plt.savefig("graphs/unlearning_prediction_vs_reel_iot.png")
    plt.close()

def remove_application_type(df, app_type="IoT_Temperature"):
    df_cleaned = df[df["Application_Type"] != app_type].copy()
    print(f"{len(df) - len(df_cleaned)} entrées supprimées pour Application_Type = '{app_type}'")
    return df_cleaned

def compare_models(model_full, model_clean, model_adv, X_test, y_test, X_test_un, y_test_un):
    from sklearn.metrics import mean_squared_error

    model_full.eval()
    model_clean.eval()
    model_adv.eval()

    with torch.no_grad():
        pred_full = model_full(X_test).numpy()
        pred_clean = model_clean(X_test_un).numpy()
        pred_adv, _ = model_adv(X_test)
        
    mse_full = mean_squared_error(y_test.numpy(), pred_full)
    mse_clean = mean_squared_error(y_test_un.numpy(), pred_clean)
    mse_adv = mean_squared_error(y_test.numpy(), pred_adv)

    print("\n Comparaison des modèles :")
    print(f"Modèle complet        (avec 'IoT_temperature') : MSE = {mse_full:.4f}")
    print(f"Modèle sans IoT       (données nettoyées)     : MSE = {mse_clean:.4f}")
    print(f"Modèle avec Unlearning (GRL)                  : MSE = {mse_adv:.4f}")

if __name__ == "__main__":
    from data_loading import load_and_prepare_data
    from anomalies import detect_anomalies
    from model_training import prepare_features, train_latency_model, evaluate_model
    

    df = load_and_prepare_data()
    df = detect_anomalies(df)

    # Modèle avant suppression
    (X_train, X_test, y_train, y_test), _ = prepare_features(df)
    model_full = train_latency_model(X_train, y_train, input_dim=X_train.shape[1])
    evaluate_model(model_full, X_test, y_test)

    # Suppression ciblée
    df_unlearned = remove_application_type(df, app_type="IoT_Temperature")
    (X_train_un, X_test_un, y_train_un, y_test_un), _ = prepare_features(df_unlearned)
    model_clean = train_latency_model(X_train_un, y_train_un, input_dim=X_train_un.shape[1])
    evaluate_model(model_clean, X_test_un, y_test_un)

    # Gradient Reversal Unlearning
    (X_train, X_test, y_train, y_test, y_train_adv, y_test_adv), encoder = prepare_data_for_unlearning(df)
    model_adv = train_unlearning_model(X_train, y_train, y_train_adv, input_dim=X_train.shape[1], num_classes=len(encoder.classes_))
    evaluate_unlearning(model_adv, X_test, y_test)

    
