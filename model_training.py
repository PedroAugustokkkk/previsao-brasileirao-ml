# Arquivo: model_training.py

import pandas as pd
import json
import joblib # Importa joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from data_collector import fetch_brasileirao_data
from feature_engineering import create_features

# Esta função será chamada pelo nosso script de build no Render
def train_and_save_all():
    """
    Função completa para buscar dados, treinar modelos e salvar os artefatos.
    """
    print("Iniciando processo de treinamento...")
    
    # 1. Buscar dados de treino
    df_raw_data = fetch_brasileirao_data(season=2025, status="FT") # FT = Finished
    
    if df_raw_data is None or df_raw_data.empty:
        print("Falha ao buscar dados de treino. Abortando.")
        return

    # 2. Engenharia de Features
    df_ml_ready = create_features(df_jogos=df_raw_data)
    
    if df_ml_ready is None or df_ml_ready.empty:
        print("Falha na engenharia de features. Abortando.")
        return

    # --- 3. Separação de Features (X) e Alvo (y) ---
    X = df_ml_ready.drop(columns=['time_casa', 'time_visitante', 'data', 'gols_casa', 'gols_visitante', 'resultado', 'pontos_casa', 'pontos_visitante'])
    y = df_ml_ready['resultado']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dados divididos em {len(X_train)} amostras de treino e {len(X_test)} de teste.")

    # --- 4. Dicionário para salvar os dados de performance ---
    performance_data = {
        "accuracyData": [],
        "classificationReport": []
    }

    # --- Modelo 1: Regressão Logística ---
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    predictions_log_reg = log_reg.predict(X_test)
    accuracy_log_reg = accuracy_score(y_test, predictions_log_reg)
    performance_data["accuracyData"].append({"model": "Reg. Logística", "accuracy": round(accuracy_log_reg * 100, 2)})

    # --- Modelo 2: Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    predictions_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, predictions_rf)
    performance_data["accuracyData"].append({"model": "Random Forest", "accuracy": round(accuracy_rf * 100, 2)})

    # --- Modelo 3: LightGBM ---
    lgbm = lgb.LGBMClassifier(random_state=42)
    lgbm.fit(X_train, y_train)
    predictions_lgbm = lgbm.predict(X_test)
    accuracy_lgbm = accuracy_score(y_test, predictions_lgbm)
    performance_data["accuracyData"].append({"model": "LightGBM", "accuracy": round(accuracy_lgbm * 100, 2)})

    # --- 5. Salvar o Modelo e os Relatórios ---
    
    # Salva o modelo LightGBM
    model_filename = "modelo_predifute.pkl"
    joblib.dump(lgbm, model_filename)
    print(f"Modelo salvo com sucesso como '{model_filename}'")

    # Gera o relatório de classificação para o LightGBM
    report = classification_report(y_test, predictions_lgbm, output_dict=True, zero_division=0)
    classes = y.unique()
    
    for cls in classes:
        if cls in report:
            performance_data["classificationReport"].append({
                "classe": cls,
                "precision": f"{report[cls]['precision']:.2f}",
                "recall": f"{report[cls]['recall']:.2f}",
                "f1Score": f"{report[cls]['f1-score']:.2f}"
            })
            
    # Salva os dados de performance em um arquivo JSON
    json_filename = "model_performance.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, ensure_ascii=False, indent=2)
    print(f"Dados de performance salvos com sucesso como '{json_filename}'")

# Esta parte permite que o script seja chamado diretamente
if __name__ == "__main__":
    train_and_save_all() # Roda o processo de treino