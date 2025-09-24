# Arquivo: feature_engineering.py

# Importações necessárias para esta tarefa.
import pandas as pd
import numpy as np

def create_features(df_jogos: pd.DataFrame):
    """
    Recebe um DataFrame de jogos e retorna um novo DataFrame com features de ML.

    Args:
        df_jogos (pd.DataFrame): O DataFrame com os dados brutos dos jogos.

    Returns:
        pd.DataFrame: Um DataFrame pronto para ser usado no treinamento do modelo.
    """
    if df_jogos is None or df_jogos.empty:
        # Se o DataFrame de entrada estiver vazio ou for nulo, retorna None.
        print("DataFrame de entrada está vazio. Nenhuma feature foi criada.")
        return None

    # --- 1. Preparação Inicial ---
    df_features = df_jogos.copy() # Copia para não alterar o original.
    df_features.dropna(subset=['gols_casa', 'gols_visitante'], inplace=True) # Remove jogos futuros.
    df_features['data'] = pd.to_datetime(df_features['data']) # Converte a coluna de data.
    df_features.sort_values(by='data', inplace=True) # Ordena por data.

    # --- 2. Criação da Coluna Alvo (Target) e Pontos ---
    df_features['resultado'] = np.where(df_features['gols_casa'] > df_features['gols_visitante'], 'VITORIA_CASA',
                                        np.where(df_features['gols_casa'] < df_features['gols_visitante'], 'VITORIA_VISITANTE', 'EMPATE'))
    
    df_features['pontos_casa'] = np.where(df_features['resultado'] == 'VITORIA_CASA', 3, np.where(df_features['resultado'] == 'EMPATE', 1, 0))
    df_features['pontos_visitante'] = np.where(df_features['resultado'] == 'VITORIA_VISITANTE', 3, np.where(df_features['resultado'] == 'EMPATE', 1, 0))

    # --- 3. Lógica para Média Móvel (Rolling Average) ---
    times = pd.unique(df_features[['time_casa', 'time_visitante']].values.ravel('K'))
    estatisticas_times = {time: [] for time in times}

    for index, row in df_features.iterrows():
        # Adiciona as estatísticas da partida atual ao histórico de cada time.
        estatisticas_times[row['time_casa']].append({'pontos': row['pontos_casa'], 'gols_marcados': row['gols_casa'], 'gols_sofridos': row['gols_visitante']})
        estatisticas_times[row['time_visitante']].append({'pontos': row['pontos_visitante'], 'gols_marcados': row['gols_visitante'], 'gols_sofridos': row['gols_casa']})

    features_calculadas = []
    JANELA = 5 # Janela de 5 jogos para calcular a 'forma'

    for index, row in df_features.iterrows():
        # Lógica de cálculo da média móvel (exatamente como no código anterior)
        hist_casa = pd.DataFrame(estatisticas_times[row['time_casa']])
        hist_visitante = pd.DataFrame(estatisticas_times[row['time_visitante']])
        idx_jogo_atual_casa = len(hist_casa) - 1
        idx_jogo_atual_visitante = len(hist_visitante) - 1
        
        media_casa = hist_casa.shift(1).rolling(window=JANELA, min_periods=1).mean().iloc[idx_jogo_atual_casa]
        media_visitante = hist_visitante.shift(1).rolling(window=JANELA, min_periods=1).mean().iloc[idx_jogo_atual_visitante]
        
        features_calculadas.append({
            'mm_pontos_casa': media_casa['pontos'],
            'mm_gols_marcados_casa': media_casa['gols_marcados'],
            'mm_gols_sofridos_casa': media_casa['gols_sofridos'],
            'mm_pontos_visitante': media_visitante['pontos'],
            'mm_gols_marcados_visitante': media_visitante['gols_marcados'],
            'mm_gols_sofridos_visitante': media_visitante['gols_sofridos']
        })

    df_novas_features = pd.DataFrame(features_calculadas, index=df_features.index)
    df_final = pd.concat([df_features, df_novas_features], axis=1)
    df_final.dropna(inplace=True) # Remove linhas onde não foi possível calcular a média.
    
    print(f"Engenharia de features concluída. DataFrame final com {len(df_final)} jogos e {len(df_final.columns)} colunas.")
    return df_final # Retorna o DataFrame final e pronto para o ML.
