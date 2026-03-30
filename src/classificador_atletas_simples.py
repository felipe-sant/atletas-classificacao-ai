import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

CAMINHO_CSV         = "AmostraDados-ABP-5DSM-Expandida.csv"
CAMINHO_MODELO      = "modelo_atletas.pkl"       # salva o modelo treinado
CAMINHO_NORMALIZADOR = "normalizador_atletas.pkl" # salva o normalizador

NOME_CLASSES = {0: "Baixo Desempenho", 1: "Médio Desempenho", 2: "Alto Desempenho"}

# Pesos de cada feature para calcular o escore de desempenho (auto-rotulação)
# Features mais importantes têm peso maior
PESOS_FEATURES = {
    "posicao_numerica"            : 0.0,  # posição é contexto, não desempenho
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


# =============================================================================
# 1. CARREGAR E PREPARAR OS DADOS
# =============================================================================

tabela = pd.read_csv(CAMINHO_CSV)

# Preenche atletas sem posição
tabela["Group"] = tabela["Group"].fillna("Indefinido")

# Converte a posição (texto) para número — a rede só entende números
mapa_posicoes = {pos: i for i, pos in enumerate(["10s", "CBs", "CMs", "STs", "WBs", "Indefinido"])}
tabela["posicao_numerica"] = tabela["Group"].map(mapa_posicoes)

# Colunas que entram na rede
colunas_features = list(PESOS_FEATURES.keys())

dados = tabela[colunas_features].values.astype(float)


# =============================================================================
# 2. NORMALIZAÇÃO
# =============================================================================
# O MinMaxScaler do sklearn faz exatamente o que precisamos:
# aprende o mínimo e máximo no treino e aplica a mesma escala depois.

normalizador = MinMaxScaler()
dados_normalizados = normalizador.fit_transform(dados)

# Salva para reutilizar na predição de atletas novos
joblib.dump(normalizador, CAMINHO_NORMALIZADOR)
print(f"Normalizador salvo em '{CAMINHO_NORMALIZADOR}'")


# =============================================================================
# 3. GERAR RÓTULOS AUTOMÁTICOS (quem é Alto, Médio, Baixo)
# =============================================================================
# Calcula um escore composto para cada atleta:
# média ponderada das features normalizadas, usando PESOS_FEATURES.

vetor_pesos = np.array(list(PESOS_FEATURES.values()))
escores     = (dados_normalizados * vetor_pesos).sum(axis=1) / vetor_pesos.sum()

# Divide em 3 faixas por percentil
corte_baixo = np.percentile(escores, 30)  # bottom 30% → Baixo
corte_alto  = np.percentile(escores, 65)  # top 35%    → Alto

rotulos = np.where(escores <= corte_baixo, 0,     # Baixo
          np.where(escores <= corte_alto,  1, 2)) # Médio ou Alto

print("\nDistribuição de classes:")
for classe, nome in NOME_CLASSES.items():
    print(f"  {nome}: {(rotulos == classe).sum()} atletas")


# =============================================================================
# 4. TREINAR A REDE NEURAL
# =============================================================================
# MLPClassifier = Multi-Layer Perceptron — rede neural feedforward do sklearn.
# hidden_layer_sizes=(32, 16) → duas camadas ocultas com 32 e 16 neurônios.

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

# Salva o modelo treinado
joblib.dump(modelo, CAMINHO_MODELO)
print(f"Modelo salvo em '{CAMINHO_MODELO}'")


# =============================================================================
# 5. PREDIÇÃO DE UM ATLETA NOVO
# =============================================================================
# Para usar em produção: carrega o normalizador e o modelo salvos,
# monta o dicionário com os dados do atleta e chama essa função.

def prever_atleta_novo(dados_atleta: dict) -> dict:
    """
    Recebe os dados brutos de um atleta e retorna a classificação.

    Parâmetros:
        dados_atleta: dicionário com os dados do atleta (mesmos nomes do CSV)

    Retorna:
        Dicionário com classe prevista, nome e probabilidades
    """
    # Carrega normalizador e modelo do disco (se ainda não estiver na memória)
    norm   = joblib.load(CAMINHO_NORMALIZADOR)
    modelo_ = joblib.load(CAMINHO_MODELO)

    posicao_numero = mapa_posicoes.get(dados_atleta.get("Group", "Indefinido"),
                                       mapa_posicoes["Indefinido"])

    vetor_bruto = np.array([[
        posicao_numero,
        dados_atleta["Workload"],
        dados_atleta["Sprint Distance"],
        dados_atleta["High Intensity Running"],
        dados_atleta["Top Speed"],
        dados_atleta["Accelerations"],
        dados_atleta["Decelerations"],
        dados_atleta["No. of Sprints"],
        dados_atleta["Metres per Minute"],
        dados_atleta["No. of High Intensity Events"],
        dados_atleta["Minutes Played"],
    ]])

    # Normaliza com os mesmos parâmetros do treino
    vetor_normalizado = norm.transform(vetor_bruto)

    classe_prevista  = int(modelo_.predict(vetor_normalizado)[0])
    probabilidades   = modelo_.predict_proba(vetor_normalizado)[0]

    return {
        "nome_classe"   : NOME_CLASSES[classe_prevista],
        "confianca"     : round(probabilidades[classe_prevista] * 100, 1),
        "probabilidades": {
            NOME_CLASSES[i]: round(float(p) * 100, 1)
            for i, p in enumerate(probabilidades)
        },
    }


# --- Exemplo de uso ---
atleta_novo = {
    "Group"                       : "STs",
    "Workload"                    : 8.5,
    "Sprint Distance"             : 110,
    "High Intensity Running"      : 430,
    "Top Speed"                   : 31.2,
    "Accelerations"               : 78,
    "Decelerations"               : 70,
    "No. of Sprints"              : 9,
    "Metres per Minute"           : 102,
    "No. of High Intensity Events": 38,
    "Minutes Played"              : 97,
}

print("\n--- Predição de atleta novo ---")
resultado = prever_atleta_novo(atleta_novo)
print(f"Classificação : {resultado['nome_classe']}")
print(f"Confiança     : {resultado['confianca']}%")
print("Probabilidades:")
for classe, prob in resultado["probabilidades"].items():
    barra = "█" * int(prob / 5)
    print(f"  {classe:<20}: {prob:>5.1f}%  {barra}")
