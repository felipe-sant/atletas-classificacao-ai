from func.predict import predict_atleta

atleta_novo = {
    "Group"                       : "10s",
    "Workload"                    : 9,
    "Sprint Distance"             : 35,
    "High Intensity Running"      : 0,
    "Top Speed"                   : 28.1,
    "Accelerations"               : 48,
    "Decelerations"               : 40,
    "No. of Sprints"              : 3,
    "Metres per Minute"           : 0,
    "No. of High Intensity Events": 38,
    "Minutes Played"              : 15,
}

print("\n--- Predição de atleta novo ---")
resultado = predict_atleta(atleta_novo)
print(f"Classificação : {resultado['nome_classe']}")
print(f"Confiança     : {resultado['confianca']}%")
print("Probabilidades:")
for classe, prob in resultado["probabilidades"].items():
    barra = "█" * int(prob / 5)
    print(f"  {classe:<20}: {prob:>5.1f}%  {barra}")