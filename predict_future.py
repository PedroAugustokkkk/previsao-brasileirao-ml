import pandas as pd
import numpy as np
import json
from data_collector import fetch_brasileirao_data
from feature_engineering import create_features 

def generate_future_predictions(modelo, df_historico, season):
    """
    Busca jogos futuros, calcula features para eles e salva as previsões.
    """
    
    df_future_raw = fetch_brasileirao_data(season=season, status="NS")
    
    if df_future_raw is None or df_future_raw.empty:
        print("Não foi possível buscar jogos futuros ou não há jogos agendados.")
        return

    df_future_raw['gols_casa'] = np.nan
    df_future_raw['gols_visitante'] = np.nan

    df_full_season = pd.concat([df_historico, df_future_raw], ignore_index=True)

    df_full_features = create_features(df_full_season)

    df_to_predict = df_full_features[df_full_features['gols_casa'].isna()].copy()
    
    if df_to_predict.empty:
        print("Não há jogos futuros para prever (talvez precisem de mais histórico).")
        return

    X_future = df_to_predict.drop(columns=['time_casa', 'time_visitante', 'data', 'gols_casa', 'gols_visitante', 'resultado', 'pontos_casa', 'pontos_visitante'])
    
    future_probabilities = modelo.predict_proba(X_future)
    
    classes = modelo.classes_

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
            "yellowCards": "N/A",
            "fouls": "N/A",
            "bothScore": "N/A",
            "mostLikelyScore": "N/A"
        })

    json_filename = "future_predictions.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(predictions_list, f, ensure_ascii=False, indent=2)
        
    print(f"Previsões de jogos futuros salvas em '{json_filename}'")