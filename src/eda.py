import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_distributions(df):
    os.makedirs("graphs", exist_ok=True)

    plt.figure(figsize=(8, 4))
    sns.histplot(df["Latency_ms"], bins=30, kde=True, color="skyblue")
    plt.title("Distribution de la Latence (ms)")
    plt.xlabel("Latence")
    plt.ylabel("Nombre d'observations")
    plt.tight_layout()
    plt.savefig("graphs/latence_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(df["Signal_dBm"], bins=30, kde=True, color="salmon")
    plt.title("Distribution du Signal (dBm)")
    plt.xlabel("Signal (dBm)")
    plt.ylabel("Nombre d'observations")
    plt.tight_layout()
    plt.savefig("graphs/signal_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.scatterplot(x="Required_BW_Mbps", y="Allocated_BW_Mbps", data=df)
    plt.title("Bande passante : Requise vs Allouée")
    plt.xlabel("Requise (Mbps)")
    plt.ylabel("Allouée (Mbps)")
    plt.tight_layout()
    plt.savefig("graphs/bande_passante.png")
    plt.close()

def plot_correlation_matrix(df):
    cols_qos = ["Latency_ms", "Signal_dBm", "Required_BW_Mbps", "Allocated_BW_Mbps"]
    correlation_matrix = df[cols_qos].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap des corrélations entre variables QoS")
    plt.tight_layout()
    plt.savefig("graphs/correlation_heatmap.png")
    plt.close()

if __name__ == "__main__":
    from data_loading import load_and_prepare_data
    df = load_and_prepare_data()
    plot_distributions(df)
    plot_correlation_matrix(df)
    print("Graphiques enregistrés dans le dossier 'graphs'")
