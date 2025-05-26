import pandas as pd
import os

def load_and_prepare_data(filepath="data/Quality of Service 5G.csv"):
    # Vérifie que le fichier existe
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} est introuvable.")

    # Chargement du fichier CSV
    df = pd.read_csv(filepath)
    print(f" Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Conversion des colonnes de texte en valeurs numériques
    df["Latency_ms"] = df["Latency"].str.replace(" ms", "").astype(float)
    df["Signal_dBm"] = df["Signal_Strength"].str.replace(" dBm", "").astype(float)

    # Conversion des bandes passantes
    def convert_bw(x):
        if pd.isna(x): return 0.0
        if "Kbps" in x:
            return float(x.replace(" Kbps", "")) / 1000
        if "Mbps" in x:
            return float(x.replace(" Mbps", ""))
        return 0.0

    df["Required_BW_Mbps"] = df["Required_Bandwidth"].apply(convert_bw)
    df["Allocated_BW_Mbps"] = df["Allocated_Bandwidth"].apply(convert_bw)

    # Nettoyage des colonnes inutiles
    df.drop(columns=["Latency", "Signal_Strength", "Required_Bandwidth", "Allocated_Bandwidth"], inplace=True)

    return df

if __name__ == "__main__":
    df = load_and_prepare_data()
    print(df.head())
