# Arquivo: main.py

# Importa as FUNÇÕES dos outros arquivos.
from data_collector import fetch_brasileirao_data
from feature_engineering import create_features

# --- Ponto de Entrada Principal do Programa ---
if __name__ == "__main__":
    # Define a temporada que queremos analisar.
    ANO_DA_TEMPORADA = 2025

    # --- ETAPA 1: Coleta de Dados ---
    # Chama a função do 'data_collector' para buscar os dados brutos.
    df_raw_data = fetch_brasileirao_data(season=ANO_DA_TEMPORADA)

    # --- ETAPA 2: Engenharia de Features ---
    # Verifica se a coleta de dados foi bem-sucedida antes de continuar.
    if df_raw_data is not None:
        # Passa os dados brutos para a função do 'feature_engineering'.
        df_ml_ready = create_features(df_jogos=df_raw_data)
        
        # --- ETAPA 3: Exibição dos Resultados ---
        # Verifica se a criação de features funcionou.
        if df_ml_ready is not None:
            print("\n--- Amostra do DataFrame Final Pronto para o Machine Learning ---")
            
            # Define as colunas que queremos visualizar.
            colunas_para_exibir = [
                'time_casa', 'time_visitante',
                'mm_pontos_casa', 'mm_gols_marcados_casa', 'mm_gols_sofridos_casa',
                'mm_pontos_visitante', 'mm_gols_marcados_visitante', 'mm_gols_sofridos_visitante',
                'resultado'
            ]
            # Exibe as primeiras 5 linhas do resultado final.
            print(df_ml_ready[colunas_para_exibir].head())
            
            # Daqui para frente, você chamaria a próxima etapa: treinamento do modelo.
            # ex: model = train_model(df_ml_ready)
