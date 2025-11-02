# Arquivo: api.py

import joblib # Para carregar o modelo .pkl
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Para o CORS
from pydantic import BaseModel
from typing import List

# Importa nossas funções de lógica de dados
from data_collector import fetch_brasileirao_data
from feature_engineering import create_features

# --- Carregamento dos Artefatos ---
MODEL_PATH = "modelo_predifute.pkl"
PERFORMANCE_PATH = "model_performance.json"

modelo = None
performance_data = None

try:
    # Carrega o modelo treinado (que o build.sh criou)
    modelo = joblib.load(MODEL_PATH)
    print("Modelo de ML carregado com sucesso.")
except FileNotFoundError:
    print(f"AVISO: Arquivo do modelo '{MODEL_PATH}' não encontrado.")
    
try:
    # Carrega o JSON de performance (que o build.sh criou)
    with open(PERFORMANCE_PATH, 'r', encoding='utf-8') as f:
        performance_data = json.load(f)
    print("Dados de performance carregados com sucesso.")
except FileNotFoundError:
    print(f"AVISO: Arquivo de performance '{PERFORMANCE_PATH}' não encontrado.")

# --- Inicialização da API ---
app = FastAPI()

# Configura o CORS
# Isso permite que o seu app na Vercel (ex: "predifute.vercel.app")
# faça requisições para sua API no Render.
origins = [
    "http://localhost:8080", # Para seu desenvolvimento local do front
    "http://localhost:5173", # Outra porta comum do Vite
    "https://predifute-brasileirao-dashboard-SEU_HASH.vercel.app" # URL do seu deploy na Vercel
    "*" # Para testes, permite tudo. Na produção, troque pelas URLs acima.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Lista de origens permitidas
    allow_credentials=True,
    allow_methods=["*"], # Permite todos os métodos (GET, POST, etc)
    allow_headers=["*"], # Permite todos os cabeçalhos
)


# --- Endpoints da API ---

@app.get("/api/performance")
def get_performance():
    """
    Endpoint que retorna os dados de performance do modelo
    (para a página "Desempenho dos Modelos")
    """
    if performance_data:
        return performance_data
    raise HTTPException(status_code=404, detail="Dados de performance ainda não gerados.")

@app.get("/api/predictions")
def get_predictions():
    """
    Endpoint que busca jogos futuros, calcula as features
    e retorna as previsões do modelo
    (para a página "Painel de Previsões")
    """
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não está pronto. Tente novamente em alguns minutos.")

    try:
        # 1. Busca dados históricos para calcular a "forma"
        df_historico = fetch_brasileirao_data(season=2025, status="FT")
        # 2. Busca jogos futuros
        df_future_raw = fetch_brasileirao_data(season=2025, status="NS")
        
        if df_future_raw is None or df_future_raw.empty:
            return [] # Retorna lista vazia se não houver jogos

        # 3. Prepara dados para engenharia de features
        df_future_raw['gols_casa'] = np.nan
        df_future_raw['gols_visitante'] = np.nan
        df_full_season = pd.concat([df_historico, df_future_raw], ignore_index=True)

        # 4. Roda a engenharia de features
        df_full_features = create_features(df_full_season)

        # 5. Isola jogos futuros com features
        df_to_predict = df_full_features[df_full_features['gols_casa'].isna()].copy()
        
        if df_to_predict.empty:
            return [] # Retorna lista vazia

        # 6. Faz as Previsões
        X_future = df_to_predict.drop(columns=['time_casa', 'time_visitante', 'data', 'gols_casa', 'gols_visitante', 'resultado', 'pontos_casa', 'pontos_visitante'])
        future_probabilities = modelo.predict_proba(X_future)
        classes = modelo.classes_

        # 7. Formata o JSON para o Front-end
        predictions_list = []
        for index, (idx_row, row) in enumerate(df_to_predict.iterrows()):
            probs = future_probabilities[index]
            prob_dict = {classe: round(prob * 100, 2) for classe, prob in zip(classes, probs)}

            predictions_list.append({
                "id": index,
                "homeTeam": row['time_casa'],
                "awayTeam": row['time_visitante'],
                "stadium": "Estádio (API não informa)",
                "dateTime": pd.to_datetime(row['data']).strftime('%d/%m/%Y %H:%M'),
                "homeWinProb": prob_dict.get('VITORIA_CASA', 0),
                "drawProb": prob_dict.get('EMPATE', 0),
                "awayWinProb": prob_dict.get('VITORIA_VISITANTE', 0),
                "yellowCards": "N/A", # Placeholder
                "fouls": "N/A", # Placeholder
                "bothScore": "N/A", # Placeholder
                "mostLikelyScore": "N/A" # Placeholder
            })
            
        return predictions_list
    
    except Exception as e:
        print(f"Erro ao gerar previsões: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar previsões: {e}")