import matplotlib
matplotlib.use('Agg')  # Empêche l'usage de Tkinter
import matplotlib.pyplot as plt
import os

# MSE réels
model_names = ["Complet", "Nettoyé (sans IoT)", "GRL (Unlearning)"]
mse_values = [308.4881, 12.3643, 151.4719]

os.makedirs("graphs", exist_ok=True)

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, mse_values)
plt.ylabel("Erreur quadratique moyenne (MSE)")
plt.title("Comparaison des modèles IA (MSE)")
plt.grid(axis='y')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.1f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("graphs/comparaison_mse_models.png")
