import pandas as pd
from math import *
import numpy as np

# Chemin automatique créé par Kaggle quand le dataset est attaché
df = pd.read_csv('/kaggle/input/predict-calorie-expenditure/train.csv')

print(df.columns)
print(df.head(2))

# Normalisation min-max pour Duration et Weight
df["Duration_norm"] = (df["Duration"] - df["Duration"].min()) / (df["Duration"].max() - df["Duration"].min())
df["Weight_norm"] = (df["Weight"] - df["Weight"].min()) / (df["Weight"].max() - df["Weight"].min())

X = np.array(df["Duration_norm"] * df["Weight_norm"])
Y = np.array(df["Calories"])

# Coefficients pour polynôme : a*X^2 + b*X + c car relation polynomiale est meilleur que lineaire
a = 0.5  # coefficient pour X^2
b = 3    # coefficient pour X
c = 1    # terme constant
learning_rate = 0.01

for y in range(5000):
    # prediction polynomiale
    y_predit = a*(X**2) + b*X + c
    ecart = y_predit - Y

    # calcule de l'erreur moyenne
    cout = sqrt(np.sum(ecart**2) / len(df))

    # mise a jour des poids
    a = a - learning_rate * 2 * np.sum(ecart * (X**2)) / len(df) # utilisation de np.sum pour bien comprendre
    b = b - learning_rate * 2 * np.sum(ecart * X) / len(df)
    c = c - learning_rate * 2 * np.sum(ecart) / len(df)

print("coeficient a : ", a," coeficient b : ", b, " coeficient c : ", c)
print(" une erreur moyenne de : ", cout)

# exemple reel
durée = int(input("entrer votre temp d effort en minute : "))
poids = int(input("entrer votre poid en kilogramme : "))

# normalisation des exemple reel avec formule : (X - X_min) / (X_max - X_min)
durée = (durée - df["Duration"].min()) / (df["Duration"].max() - df["Duration"].min())
poids = ( poids - df["Weight"].min()) / (df["Weight"].max() - df["Weight"].min())

X_exemple = durée * poids
calories_depenser = a * X_exemple **2 + b * X_exemple + c

print()
print("vous depenserais entre : ", calories_depenser - cout ," et ", calories_depenser + cout , " calories. Bien que normalement proche de ", calories_depenser , " calories " )


# visualisation faite par chat gpt (flemme)
import matplotlib.pyplot as plt
import numpy as np

# points réels
plt.scatter(X, df["Calories"], color='blue', alpha=0.6, label="Données réelles")

# courbe polynomiale lisse
X_plot = np.linspace(min(X), max(X), 100)  # 100 points pour une belle courbe
Y_plot = a * X_plot**2 + b * X_plot + c
plt.plot(X_plot, Y_plot, color='red', linewidth=2, label="Modèle polynomiale")

plt.xlabel("Durée de l'effort * poids normalisé")
plt.ylabel("Calories dépensées")
plt.title("Relation Effort vs Calorie (polynôme aX^2 + bX + c)")
plt.legend()
plt.grid(True)  # optionnel pour mieux visualiser
plt.show()