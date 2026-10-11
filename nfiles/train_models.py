import time
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from data_utils import read_dataset
from plot_experiments import get_eff_pos
from data_split import create_spatial_split, VOXEL_SIZE, TEST_SIZE, N_FOLDS, compute_voxel_volume
from features import build_features
FILENAME = "dataset_regressao_energia.csv"


# ==============================================================================
# 2. DEFINIÇÃO DA GRADE DE MODELOS (DO MAIS BÁSICO AO AVANÇADO)
# ==============================================================================
def get_modelos():
    return {
        # 1. Baseline trivial
        "01. Dummy (Média)": DummyRegressor(strategy="mean"),
        
        # 2. Modelos lineares clássicos
        "02. Regressão Linear Simples": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ]),
        "03. Ridge Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=10.0))
        ]),
        "04. Lasso Regression (L1)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Lasso(alpha=0.1, max_iter=2000))
        ]),
        
        # 3. Linear com interações físicas (grau 2)
        "05. Polinomial Grau 2 + Ridge": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=50.0))
        ]),
        
        # 4. Modelos não lineares intermediários
        "06. Decision Tree": DecisionTreeRegressor(max_depth=7, min_samples_leaf=5, random_state=42),
        "07. SVR (RBF Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(C=50.0, epsilon=0.5))
        ]),
        "08. MLP Regressor (Rede Neural)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42, early_stopping=True))
        ]),
        
        # 5. Ensembles avançados
        "09. Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1),
        "10. Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.07, random_state=42)
    }


# ==============================================================================
# 3. ROTINA DE TREINAMENTO E VALIDAÇÃO CRUZADA
# ==============================================================================
def executar_benchmark():
    print("--- 1. Carregando dados e gerando splits espaciais ---")
    X, y = read_dataset(FILENAME)
    pos_exp = get_eff_pos(X, y)
    
    # Gera partições espaciais por cubos
    splits = create_spatial_split(
        X, y, pos_exp,
        voxel_size=VOXEL_SIZE,
        test_size=TEST_SIZE,
        n_folds=N_FOLDS,
        random_state=42
    )

    metrics_voxel = compute_voxel_volume(pos_exp)

    
    idx_dev = splits["idx_dev"]
    idx_test = splits["idx_test"]
    folds = splits["folds"]
    
    print("\n--- 2. Enriquecimento de atributos físicos ---")
    X_exp = build_features(X)
    
    modelos = get_modelos()
    tabela_metricas = []
    modelos_treinados = {}
    
    print(f"\n--- 3. Executando K-Fold ({N_FOLDS} dobras) e Teste Cego ---")
    for nome, modelo_base in modelos.items():
        r2_folds, rmse_folds, mae_folds = [], [], []
        t0 = time.time()
        
        # Loop nas K dobras de desenvolvimento
        for f in folds:
            tr_idx = f["train_idx"]
            val_idx = f["val_idx"]
            
            X_tr, y_tr = X_exp.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X_exp.iloc[val_idx], y.iloc[val_idx]
            
            clf = clone(modelo_base)
            clf.fit(X_tr, y_tr)
            
            preds_va = clf.predict(X_va)
            r2_folds.append(r2_score(y_va, preds_va))
            rmse_folds.append(np.sqrt(mean_squared_error(y_va, preds_va)))
            mae_folds.append(mean_absolute_error(y_va, preds_va))
            
        tempo_cv = time.time() - t0
        
        # Ajuste final em todo o desenvolvimento para avaliação no Teste Cego
        modelo_final = clone(modelo_base)
        modelo_final.fit(X_exp.iloc[idx_dev], y.iloc[idx_dev])
        modelos_treinados[nome] = modelo_final
        
        preds_test = modelo_final.predict(X_exp.iloc[idx_test])
        test_r2 = r2_score(y.iloc[idx_test], preds_test)
        test_rmse = np.sqrt(mean_squared_error(y.iloc[idx_test], preds_test))
        test_mae = mean_absolute_error(y.iloc[idx_test], preds_test)
        
        tabela_metricas.append({
            "Modelo": nome,
            "CV R² (Méd ± Desv)": f"{np.mean(r2_folds):.4f} ± {np.std(r2_folds):.4f}",
            "CV RMSE (J)": f"{np.mean(rmse_folds):.2f} ± {np.std(rmse_folds):.2f}",
            "CV MAE (J)": f"{np.mean(mae_folds):.2f} ± {np.std(mae_folds):.2f}",
            "Teste R²": round(test_r2, 4),
            "Teste RMSE (J)": round(test_rmse, 3),
            "Teste MAE (J)": round(test_mae, 3),
            "Tempo (s)": round(tempo_cv, 2)
        })
        print(f"✓ {nome:<30} | Teste R²: {test_r2:.4f} | Teste RMSE: {test_rmse:.2f} J")

    # Consolidação em DataFrame
    df_res = pd.DataFrame(tabela_metricas).sort_values("Teste R²", ascending=False)
    
    print("\n" + "=" * 90)
    print("TABELA COMPARATIVA DE DESEMPENHO DOS MODELOS")
    print("=" * 90)
    print(df_res.to_string(index=False))
    

    # 4. Exportação do melhor modelo para uso no PyMOO
    melhor_nome = df_res.iloc[0]["Modelo"]
    melhor_objeto = modelos_treinados[melhor_nome]
    nome_arquivo_pkl = "melhor_modelo_energia.pkl"
    joblib.dump(melhor_objeto, nome_arquivo_pkl)

    print("\n" + "=" * 90)
    print(f"Melhor Modelo: {melhor_nome}")
    print(f"Exportado para '{nome_arquivo_pkl}' para integração com o otimizador PyMOO.")
    print("=" * 90)

    # Previsão final do melhor modelo no conjunto de teste cego
    y_test = y.iloc[idx_test]
    y_pred_best = melhor_objeto.predict(X_exp.iloc[idx_test])

    return {
        "df_res": df_res,
        "melhor_modelo": melhor_objeto,
        "melhor_nome": melhor_nome,
        "y_test": y_test,
        "y_pred_best": y_pred_best
    }


