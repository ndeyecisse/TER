import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def detect_anomalies(df):
    seuil_latence = 80        
    seuil_signal = -70          

    df["Anomalie_Latence"] = df["Latency_ms"] > seuil_latence
    df["Anomalie_Signal"] = df["Signal_dBm"] < seuil_signal
    df["Anomalie_BW"] = df["Allocated_BW_Mbps"] < df["Required_BW_Mbps"]
    df["Anomalie_Global"] = df[["Anomalie_Latence", "Anomalie_Signal", "Anomalie_BW"]].any(axis=1)

    def classifier_anomalie(row):
        if row["Anomalie_Latence"] and row["Anomalie_Signal"] and row["Anomalie_BW"]:
            return "3. Latence + Signal + BW"
        elif row["Anomalie_Latence"] and row["Anomalie_Signal"]:
            return "2. Latence + Signal"
        elif row["Anomalie_Latence"] and row["Anomalie_BW"]:
            return "2. Latence + BW"
        elif row["Anomalie_Signal"] and row["Anomalie_BW"]:
            return "2. Signal + BW"
        elif row["Anomalie_Latence"]:
            return "1. Latence"
        elif row["Anomalie_Signal"]:
            return "1. Signal"
        elif row["Anomalie_BW"]:
            return "1. BW"
        else:
            return "0. Normal"

    df["Type_Anomalie"] = df.apply(classifier_anomalie, axis=1)



    return df

def export_anomalies(df, path="reports/anomalies_detectees.csv"):
    os.makedirs("reports", exist_ok=True)
    df[df["Anomalie_Global"] == True].to_csv(path, index=False)
    print(f"Anomalies enregistrées dans '{path}'")

def plot_anomalies(df):
    os.makedirs("graphs", exist_ok=True)

    # Bande passante
    plt.figure(figsize=(8, 4))
    df["Anomalie_BW"] = df["Anomalie_BW"].map({True: "Anomalie", False: "Normal"})
    palette = {"Normal": "green", "Anomalie": "red"}
    sns.scatterplot(
    x="Required_BW_Mbps", 
    y="Allocated_BW_Mbps", 
    hue="Anomalie_BW", 
    data=df, 
    palette=palette
    )
    plt.title("Anomalies de bande passante (Requise vs Allouée)")
    plt.xlabel("Requise (Mbps)")
    plt.ylabel("Allouée (Mbps)")
    plt.legend(title="Anomalie BW")
    plt.tight_layout() 
    plt.savefig("graphs/anomalie_bw.png")
    plt.close()

    # Latence
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="Latency_ms", hue="Anomalie_Latence", bins=30, kde=True, palette=["green", "red"])
    plt.title("Distribution de la latence avec anomalies")
    plt.xlabel("Latence (ms)")
    plt.tight_layout()
    plt.savefig("graphs/anomalie_latence.png")
    plt.close()

    # Signal
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="Signal_dBm", hue="Anomalie_Signal", bins=30, kde=True, palette=["green", "red"])
    plt.title("Distribution du signal avec anomalies")
    plt.xlabel("Signal (dBm)")
    plt.tight_layout()
    plt.savefig("graphs/anomalie_signal.png")
    plt.close()

    # Vue globale
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Required_BW_Mbps",
        y="Allocated_BW_Mbps",
        hue="Type_Anomalie",
        data=df,
        palette="Set2"
    )
    plt.title("Anomalies globales : Requise vs Allouée")
    plt.xlabel("Requise (Mbps)")
    plt.ylabel("Allouée (Mbps)")
    plt.legend(title="Type d'anomalie")
    plt.tight_layout()
    plt.savefig("graphs/anomalie_global_view.png")
    plt.close()


if __name__ == "__main__":
    from data_loading import load_and_prepare_data
    df = load_and_prepare_data()
    df = detect_anomalies(df)
    print(f"\nTaux global d'anomalies : {df['Anomalie_Global'].mean():.2%}")
    export_anomalies(df)
    plot_anomalies(df)