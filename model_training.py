import pandas as pd
import json 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


def train_and_evaluate_model(df_ml_ready: pd.DataFrame):
    if df_ml_ready is None or df_ml_ready.empty:
        print("DataFrame para treino está vazio. Abortando treinamento.")
        return

    X = df_ml_ready.drop(columns=['time_casa', 'time_visitante', 'data', 'gols_casa', 'gols_visitante', 'resultado', 'pontos_casa', 'pontos_visitante'])
    y = df_ml_ready['resultado']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dados divididos em {len(X_train)} amostras de treino e {len(X_test)} de teste.")

    performance_data = {
        "accuracyData": [],
        "classificationReport": []
    }

    print("\n--- Treinando Modelo Baseline: Regressão Logística ---")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    predictions_log_reg = log_reg.predict(X_test)
    accuracy_log_reg = accuracy_score(y_test, predictions_log_reg)
    print(f"Acurácia da Regressão Logística: {accuracy_log_reg * 100:.2f}%")
    performance_data["accuracyData"].append({"model": "Reg. Logística", "accuracy": round(accuracy_log_reg * 100, 2)})

    print("\n--- Treinando Modelo Padrão: Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    predictions_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, predictions_rf)
    print(f"Acurácia do Random Forest: {accuracy_rf * 100:.2f}%")
    performance_data["accuracyData"].append({"model": "Random Forest", "accuracy": round(accuracy_rf * 100, 2)})

    print("\n--- Treinando Modelo Avançado: LightGBM ---")
    lgbm = lgb.LGBMClassifier(random_state=42)
    lgbm.fit(X_train, y_train)
    predictions_lgbm = lgbm.predict(X_test)
    accuracy_lgbm = accuracy_score(y_test, predictions_lgbm)
    print(f"Acurácia do LightGBM: {accuracy_lgbm * 100:.2f}%")
    performance_data["accuracyData"].append({"model": "LightGBM", "accuracy": round(accuracy_lgbm * 100, 2)})

    model_filename = "modelo_predifute.pkl"
    joblib.dump(lgbm, model_filename)
    print(f"\nModelo salvo com sucesso como '{model_filename}'")

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
            
    json_filename = "model_performance.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, ensure_ascii=False, indent=2) 
    print(f"Dados de performance salvos com sucesso como '{json_filename}'")
    
    return lgbm