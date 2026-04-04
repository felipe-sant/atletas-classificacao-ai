import numpy as np
import joblib

CAMINHO_MODELO      = "./model/modelo_atletas.pkl"       
CAMINHO_NORMALIZADOR = "./model/normalizador_atletas.pkl" 
NOME_CLASSES = {0: "Baixo Desempenho", 1: "Médio Desempenho", 2: "Alto Desempenho"}

mapa_posicoes = {pos: i for i, pos in enumerate(["10s", "CBs", "CMs", "STs", "WBs", "Indefinido"])}

def predict_atleta(dados_atleta: dict) -> dict:
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