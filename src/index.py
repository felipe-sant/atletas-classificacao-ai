import pandas as pd

path_data = "./data/AmostraDados-ABP-5DSM.csv"
# path_data = "./data/AmostraDados-ABP-5DSM-Expandida.csv"

df = pd.read_csv(path_data)

# normalizando

features = [ 
    "Workload",
    "Sprint Distance",
    "High Intensity Running",
    "Top Speed",
    "Accelerations",
    "Decelerations",
    "No. of Sprints",
    "Metres per Minute",
    "No. of High Intensity Events"
]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# aplicando os pesos

weights = {
    "Workload": 0.8,
    "Sprint Distance": 1.2,
    "High Intensity Running": 1.2,
    "Top Speed": 1.5,
    "Accelerations": 1.1,
    "Decelerations": 1.0,
    "No. of Sprints": 1.2,
    "Metres per Minute": 1.3,
    "No. of High Intensity Events": 1.1
}
for col in features:
    df[col] = df[col] * weights[col]

# adicionando o resultado pra IA poder usar como base pro treinamento entre 0-1

df["raw_score"] = df[features].sum(axis=1)

scores = []
for group, gdf in df.groupby("Group"):
    gdf = gdf.copy()

    min_val = gdf["raw_score"].min()
    max_val = gdf["raw_score"].max()

    gdf["target_score"] = (gdf["raw_score"] - min_val) / (max_val - min_val)

    scores.append(gdf)
df = pd.concat(scores)

# preparar dados pra IA

X = df[features + ["Group"]]
X = pd.get_dummies(X, columns=["Group"])
Y = df["target_score"]

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(X, y, epochs=100, batch_size=16)

model.save("./model/modelo_jogador.keras")