import pandas as pd

def read_dataset(filename):
    df = pd.read_csv(filename)

    # selecao de variaveis de entrada e alvo de consumo
    colunas_features = []
    for i in range(1, 6):
        colunas_features += [f"servo{i}_angulo_inicial", f"servo{i}_angulo_final"]

    X = df[colunas_features].copy()
    y = df["energia_total_J_movimento"].copy()

    print(f"Dataframe carregado:")
    print(f"Amostras: {X.shape[0]}")
    print(f"features: {X.shape[1]} colunas de ângulos")
    df.head()
    return [X, y]