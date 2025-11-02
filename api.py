# -*- coding: utf-8 -*-
# Comentários em cada linha, como solicitado.

import joblib # Para carregar o modelo .pkl
import json # Para carregar o JSON de performance
import pandas as pd # Para manipulação de dados
import numpy as np # Para operações numéricas
from fastapi import FastAPI, HTTPException # O framework da API
from fastapi.middleware.cors import CORSMiddleware # Para permitir acesso do front-end
from typing import List # Para definir tipos de dados

# Importa nossas próprias funções de lógica de dados
from data_collector import fetch_brasileirao_data
from feature_engineering import create_features

# --- Definição dos caminhos dos arquivos ---
MODEL_PATH = "modelo_predifute.pkl" # Caminho para o modelo salvo
PERFORMANCE_PATH = "model_performance.json" # Caminho para o JSON de performance

# --- Carregamento dos Artefatos ---
modelo = None # Variável para guardar o modelo carregado
performance_data = None # Variável para guardar os dados de performance

try:
    # Carrega o modelo treinado (que o build.sh criou)
    modelo = joblib.load(MODEL_PATH)
    print("Modelo de ML carregado com sucesso.") # Log de sucesso
except FileNotFoundError:
    # Aviso caso o build tenha falhado e o modelo não exista
    print(f"AVISO: Arquivo do modelo '{MODEL_PATH}' não encontrado. O endpoint /api/predictions falhará.")
    
try:
    # Carrega o JSON de performance (que o build.sh criou)
    with open(PERFORMANCE_PATH, 'r', encoding='utf-8') as f:
        performance_data = json.load(f)
    print("Dados de performance carregados com sucesso.") # Log de sucesso
except FileNotFoundError:
    # Aviso caso o build tenha falhado e o JSON não exista
    print(f"AVISO: Arquivo de performance '{PERFORMANCE_PATH}' não encontrado. O endpoint /api/performance falhará.")

# --- Inicialização da API ---
app = FastAPI() # Cria a instância da aplicação FastAPI

# Configura o CORS (Cross-Origin Resource Sharing)
origins = [
    # Adicione a URL do seu site na Vercel quando souber
    "https://predifute-brasileirao-dashboard.vercel.app", # Exemplo
    "http://localhost:8080", # Para seu desenvolvimento local do front (Vite)
    "http://localhost:5173", # Outra porta comum do Vite
    "*" # Permite todas as origens (bom para testes, mas restrinja na produção)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Lista de origens permitidas
    allow_credentials=True, # Permite credenciais (cookies, etc)
    allow_methods=["*"], # Permite todos os métodos (GET, POST, etc)
    allow_headers=["*"], # Permite todos os cabeçalhos
)

# --- Endpoints da API ---

# Endpoint de "Saúde" (para verificar se a API está no ar)
@app.get("/api")
def read_root():
    return {"status": "PrediFute API no ar!"}

# Endpoint para a página "Desempenho dos Modelos"
@app.get("/api/performance")
def get_performance():
    """
    Endpoint que retorna os dados de performance do modelo
    (para a página "Desempenho dos Modelos")
    """
    if performance_data:
        return performance_data # Retorna o JSON de performance carregado
    # Se o arquivo não foi carregado, retorna um erro 404
    raise HTTPException(status_code=404, detail="Dados de performance ainda não gerados ou não encontrados.")

# Endpoint para a página "Painel de Previsões"
@app.get("/api/predictions")
def get_predictions():
    """
    Endpoint que busca jogos futuros, calcula as features
    e retorna as previsões do modelo
    (para a página "Painel de Previsões")
    """
    # Verifica se o modelo foi carregado com sucesso
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não está pronto. O build pode estar em andamento.")

    try:
        # --- Lógica de Previsão (movida do predict_future.py para cá) ---
        
        # 1. Busca dados históricos (jogos finalizados)
        df_historico = fetch_brasileirao_data(season=2025, status="FT")
        # 2. Busca jogos futuros (jogos não iniciados)
        df_future_raw = fetch_brasileirao_data(season=2025, status="NS")
        
        if df_future_raw is None or df_future_raw.empty:
            return [] # Retorna lista vazia se não houver jogos futuros

        # 3. Prepara dados para engenharia de features
        df_future_raw['gols_casa'] = np.nan # Garante que jogos futuros não têm placar
        df_future_raw['gols_visitante'] = np.nan
        # Junta o histórico com o futuro para calcular as features
        df_full_season = pd.concat([df_historico, df_future_raw], ignore_index=True)

        # 4. Roda a engenharia de features em tudo
        df_full_features = create_features(df_full_season)

        # 5. Isola apenas os jogos futuros (que agora têm features)
        df_to_predict = df_full_features[df_full_features['resultado'].isna()].copy()
        
        if df_to_predict.empty:
            return [] # Retorna lista vazia se não houver jogos para prever

        # 6. Faz as Previsões
        # Prepara o X (features) para os jogos futuros
        X_future = df_to_predict.drop(columns=['time_casa', 'time_visitante', 'data', 'gols_casa', 'gols_visitante', 'resultado', 'pontos_casa', 'pontos_visitante'])
        # Pede as PROBABILIDADES (ex: [0.45, 0.30, 0.25])
        future_probabilities = modelo.predict_proba(X_future)
        # Pega os nomes das classes (ex: 'VITORIA_CASA', 'EMPATE', 'VITORIA_VISITANTE')
        classes = modelo.classes_

        # 7. Formata o JSON para o Front-end
        predictions_list = []
        for index, (idx_row, row) in enumerate(df_to_predict.iterrows()):
            probs = future_probabilities[index] # Pega as probabilidades para este jogo
            # Cria um dicionário com as probabilidades formatadas
            prob_dict = {classe: round(prob * 100, 2) for classe, prob in zip(classes, probs)}

            # Monta o objeto JSON exatamente como o seu front-end espera
            predictions_list.append({
                "id": index, # Um ID simples
                "homeTeam": row['time_casa'],
                "awayTeam": row['time_visitante'],
                "stadium": "Estádio (API não informa)", # API-Football não fornece estádio facilmente
                "dateTime": pd.to_datetime(row['data']).strftime('%d/%m/%Y %H:%M'), # Formata a data
                "homeWinProb": prob_dict.get('VITORIA_CASA', 0),
                "drawProb": prob_dict.get('EMPATE', 0),
                "awayWinProb": prob_dict.get('VITORIA_VISITANTE', 0),
                "yellowCards": "N/A", # Placeholder (próximo passo do projeto)
                "fouls": "N/A", # Placeholder (próximo passo do projeto)
                "bothScore": "N/A", # Placeholder (próximo passo do projeto)
                "mostLikelyScore": "N/A" # Placeholder (próximo passo do projeto)
            })
            
        return predictions_list # Retorna a lista de previsões
    
    except Exception as e:
        # Captura qualquer erro inesperado durante o processo
        print(f"Erro ao gerar previsões: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar previsões: {str(e)}")
