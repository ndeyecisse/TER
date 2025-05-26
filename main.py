# === main.py : Orchestrateur du projet TER ===
from src.unlearning import compare_models
from src.data_loading import load_and_prepare_data
from src.eda import plot_distributions, plot_correlation_matrix
from src.anomalies import detect_anomalies, export_anomalies, plot_anomalies
from src.model_training import prepare_features, train_latency_model, evaluate_model
from src.unlearning import remove_application_type, prepare_data_for_unlearning, train_unlearning_model, evaluate_unlearning

import random
import numpy as np
import torch

SEED = 40
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("\n=== Étape 1 : Chargement et préparation des données ===")
df = load_and_prepare_data()

print("\n=== Étape 2 : Analyse exploratoire (EDA) ===")
plot_distributions(df)
plot_correlation_matrix(df)

print("\n=== Étape 3 : Détection d'anomalies ===")
df = detect_anomalies(df)
export_anomalies(df)
plot_anomalies(df)

# 🔍 Étape intermédiaire : Analyse des applications les plus anormales
print("\n=== Analyse : Taux d'anomalies par type d'application ===")
anomalie_stats = df.groupby("Application_Type")["Anomalie_Global"].mean().sort_values(ascending=False)
print(anomalie_stats)

print("\n=== Étape 4 : Modèle prédictif (réseau de neurones classique) ===")
(X_train, X_test, y_train, y_test), _ = prepare_features(df)
model_full = train_latency_model(X_train, y_train, input_dim=X_train.shape[1])
evaluate_model(model_full, X_test, y_test)


print("\n=== Étape 5 : Suppression ciblée (Unlearning) ===")
df_unlearned = remove_application_type(df, app_type="IoT_Temperature")
(X_train_un, X_test_un, y_train_un, y_test_un), _ = prepare_features(df_unlearned)
model_clean = train_latency_model(X_train_un, y_train_un, input_dim=X_train_un.shape[1])
evaluate_model(model_clean, X_test_un, y_test_un)

print("\n=== Étape 6 : Gradient Reversal (Unlearning) ===")
(X_train_adv, X_test_adv, y_train_adv, y_test_adv, y_train_app, y_test_app), encoder = prepare_data_for_unlearning(df)
model_adv = train_unlearning_model(X_train_adv, y_train_adv, y_train_app, input_dim=X_train_adv.shape[1], num_classes=len(encoder.classes_), lambda_=1.5)
evaluate_unlearning(model_adv, X_test_adv, y_test_adv, y_test_app)

print("\n=== Étape 7 : Comparaison des modèles ===")
compare_models(
    model_full,     # le modèle complet entraîné au début
    model_clean,    # modèle sans IoT_Temperature
    model_adv,      # modèle GRL
    X_test, y_test,
    X_test_un, y_test_un,
    y_test_app, encoder
)

print("\n Projet TER exécuté avec succès. Résultats dans 'graphs/' et 'reports/'")
