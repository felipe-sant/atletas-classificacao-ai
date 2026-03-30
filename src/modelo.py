import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# config

CAMINHO_CSV         = "AmostraDados-ABP-5DSM-Expandida.csv"
CAMINHO_MODELO      = "modelo_atletas.pkl"       
CAMINHO_NORMALIZADOR = "normalizador_atletas.pkl" 
NOME_CLASSES = {0: "Baixo Desempenho", 1: "Médio Desempenho", 2: "Alto Desempenho"}

PESOS_FEATURES = {
    "posicao_numerica"            : 0.0, 
    "Workload"                    : 1.5,
    "Sprint Distance"             : 1.5,
    "High Intensity Running"      : 1.5,
    "Top Speed"                   : 1.0,
    "Accelerations"               : 1.0,
    "Decelerations"               : 1.0,
    "No. of Sprints"              : 1.2,
    "Metres per Minute"           : 1.5,
    "No. of High Intensity Events": 1.0,
    "Minutes Played"              : 0.8,
}

# carregar dados

tabela = pd.read_csv(CAMINHO_CSV)

tabela["Group"] = tabela["Group"].fillna("Indefinido")

mapa_posicoes = {pos: i for i, pos in enumerate(["10s", "CBs", "CMs", "STs", "WBs", "Indefinido"])}
tabela["posicao_numerica"] = tabela["Group"].map(mapa_posicoes)

colunas_features = list(PESOS_FEATURES.keys())

dados = tabela[colunas_features].values.astype(float)

# normalizar

normalizador = MinMaxScaler()
dados_normalizados = normalizador.fit_transform(dados)

joblib.dump(normalizador, CAMINHO_NORMALIZADOR)
print(f"Normalizador salvo em '{CAMINHO_NORMALIZADOR}'")

# gerar rotulos

vetor_pesos = np.array(list(PESOS_FEATURES.values()))
escores     = (dados_normalizados * vetor_pesos).sum(axis=1) / vetor_pesos.sum()

corte_baixo = np.percentile(escores, 30) 
corte_alto  = np.percentile(escores, 65) 

rotulos = np.where(escores <= corte_baixo, 0,
          np.where(escores <= corte_alto,  1, 2)) 

print("\nDistribuição de classes:")
for classe, nome in NOME_CLASSES.items():
    print(f"  {nome}: {(rotulos == classe).sum()} atletas")

# treinar rede neural

modelo = MLPClassifier(
    hidden_layer_sizes = (32, 16),  # arquitetura da rede
    activation         = "relu",    # função de ativação
    max_iter           = 1500,      # número de épocas
    learning_rate_init = 0.01,      # taxa de aprendizado
    random_state       = 42,        # reprodutibilidade
    verbose            = False,
)

print("\nTreinando a rede neural...")
modelo.fit(dados_normalizados, rotulos)

acuracia = modelo.score(dados_normalizados, rotulos)
print(f"Acurácia no treino: {acuracia * 100:.1f}%")

joblib.dump(modelo, CAMINHO_MODELO)
print(f"Modelo salvo em '{CAMINHO_MODELO}'")