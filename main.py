# Arquivo: main.py

from data_collector import fetch_brasileirao_data
from feature_engineering import create_features
from model_training import train_and_evaluate_model
from predict_future import generate_future_predictions

if __name__ == "__main__":
    ANO_DA_TEMPORADA = 2025

    df_raw_data = fetch_brasileirao_data(season=ANO_DA_TEMPORADA, status="FT")
    
    if df_raw_data is not None:
        df_ml_ready = create_features(df_jogos=df_raw_data)
        
        if df_ml_ready is not None:
            modelo_treinado = train_and_evaluate_model(df_ml_ready)

            if modelo_treinado:
                print("\n--- Gerando previsões para jogos futuros ---")
                generate_future_predictions(
                    modelo=modelo_treinado, 
                    df_historico=df_raw_data, 
                    season=ANO_DA_TEMPORADA
                )