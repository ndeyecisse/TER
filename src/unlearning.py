
import torch
import torch.nn as nn
from torch.autograd import Function
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os

# === GRL : fonction d'inversion de gradient ===
def grad_reverse(x, lambda_=1.5):
    class GradientReversalFunction(Function):
        @staticmethod
        def forward(ctx, input):
            ctx.lambda_ = lambda_
            return input.view_as(input)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.lambda_

    return GradientReversalFunction.apply(x)

# === Modèle avec branche adversaire ===
class GradientUnlearningModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GradientUnlearningModel, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.adversary = nn.Sequential(
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    


    def forward(self, x, lambda_=1.5):
        shared_feat = self.shared(x)
        reverse_feat = grad_reverse(shared_feat, lambda_)
        return self.regressor(shared_feat), self.adversary(reverse_feat)

# === Préparation des données pour GRL ===
def prepare_data_for_unlearning(df):
    label_encoder = LabelEncoder()
    y_adv = label_encoder.fit_transform(df["Application_Type"])

    df_encoded = pd.get_dummies(df, columns=["Application_Type"], drop_first=True)
    features = df_encoded.drop(columns=[
        "Timestamp", "User_ID", "Latency_ms", "Resource_Allocation",
        "Anomalie_Latence", "Anomalie_Signal", "Anomalie_BW",
        "Anomalie_Global", "Type_Anomalie"
    ], errors="ignore")
    
    target = df["Latency_ms"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)
    y_adv_tensor = torch.tensor(y_adv, dtype=torch.long)

    return train_test_split(X_tensor, y_tensor, y_adv_tensor, test_size=0.2, random_state=42), label_encoder

# === Entraînement du modèle GRL ===
def train_unlearning_model(X_train, y_train, y_train_adv, input_dim, num_classes, lambda_=1.5, epochs=600):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GradientUnlearningModel(input_dim, num_classes).to(device)
    criterion_reg = nn.MSELoss()
    criterion_adv = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()

        #  Entraîner l'adversaire 17x plus souvent
        for _ in range(17):
            _, pred_app = model(X_train, lambda_=lambda_)
            loss_adv = criterion_adv(pred_app, y_train_adv)
            optimizer.zero_grad()
            loss_adv.backward()
            optimizer.step()

        #  Puis entraîner le régresseur avec GRL
        pred_lat, pred_app = model(X_train, lambda_=lambda_)
        loss_lat = criterion_reg(pred_lat, y_train)
        loss_adv = criterion_adv(pred_app, y_train_adv)
        total_loss = loss_lat + lambda_ * loss_adv

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Latency Loss: {loss_lat.item():.4f}, Adversarial Loss: {loss_adv.item():.4f}")


    return model

# === Évaluation GRL ===
def evaluate_unlearning(model, X_test, y_test, y_test_app):
    model.eval()
    with torch.no_grad():
        pred_latency, pred_app_type = model(X_test)
        predictions = pred_latency.numpy()
        true = y_test.numpy()

        pred_labels = torch.argmax(pred_app_type, dim=1)
        accuracy = accuracy_score(y_test_app.numpy(), pred_labels.numpy())

        mse_unlearn = mean_squared_error(true, predictions)

    print(f"\n MSE du modèle GRL : {mse_unlearn:.4f}")
    print(f"Accuracy adversaire (doit être faible si unlearning efficace) : {accuracy:.4f}")

    os.makedirs("graphs", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.scatter(true, predictions, alpha=0.7)
    plt.xlabel("Latence réelle (ms)")
    plt.ylabel("Latence prédite (ms)")
    plt.title("Prédiction de la latence avec GRL")
    plt.grid(True)
    plt.plot([min(true), max(true)], [min(true), max(true)], "r--")
    plt.tight_layout()
    plt.savefig("graphs/unlearning_prediction_vs_reel.png")
    plt.close()

    return mse_unlearn, accuracy, predictions, true

# === Suppression ciblée ===
def remove_application_type(df, app_type="IoT_Temperature"):
    df_cleaned = df[df["Application_Type"] != app_type].copy()
    print(f"{len(df) - len(df_cleaned)} lignes supprimées pour l'application : {app_type}")
    return df_cleaned

def compare_models(model_full, model_clean, model_adv, X_test, y_test, X_test_un, y_test_un, y_test_app, encoder):
    """
    Compare les performances des trois modèles :
    - modèle complet
    - modèle sans application supprimée
    - modèle avec unlearning GRL

    Affiche : MSE pour la latence, Accuracy adversaire
    """
    model_full.eval()
    model_clean.eval()
    model_adv.eval()

    with torch.no_grad():
        # Modèle 1 : complet
        pred = model_full(X_test)
        if isinstance(pred, tuple):
            pred_full = pred[0].numpy()
        else:
            pred_full = pred.numpy()
        mse_full = mean_squared_error(y_test.numpy(), pred_full)

        # Modèle 2 : sans IoT_Temperature
        pred_clean = model_clean(X_test_un).numpy()
        mse_clean = mean_squared_error(y_test_un.numpy(), pred_clean)

        # Modèle 3 : GRL
        pred_latency_adv, pred_app_adv = model_adv(X_test)
        mse_adv = mean_squared_error(y_test.numpy(), pred_latency_adv.numpy())

        # Accuracy adversaire
        pred_app_labels = torch.argmax(pred_app_adv, dim=1)
        accuracy_adv = accuracy_score(y_test_app.numpy(), pred_app_labels.numpy())

    print("\n=== Comparaison des modèles ===")
    print(f" Modèle Complet          → MSE = {mse_full:.4f}")
    print(f" Modèle Nettoyé (sans IoT) → MSE = {mse_clean:.4f}")
    print(f" Modèle GRL (Unlearning)   → MSE = {mse_adv:.4f}")
    print(f" Accuracy adversaire GRL   → {accuracy_adv:.4f} (doit être proche du hasard)")
