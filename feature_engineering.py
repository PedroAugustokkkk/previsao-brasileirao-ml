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
        print("DataFrame de entrada está vazio. Nenhuma feature foi criada.")
        return None

    df_features = df_jogos.copy() 
    df_features.dropna(subset=['gols_casa', 'gols_visitante'], inplace=True) 
    df_features['data'] = pd.to_datetime(df_features['data'])
    df_features.sort_values(by='data', inplace=True)

    df_features['resultado'] = np.where(df_features['gols_casa'] > df_features['gols_visitante'], 'VITORIA_CASA',
                                        np.where(df_features['gols_casa'] < df_features['gols_visitante'], 'VITORIA_VISITANTE', 'EMPATE'))
    
    df_features['pontos_casa'] = np.where(df_features['resultado'] == 'VITORIA_CASA', 3, np.where(df_features['resultado'] == 'EMPATE', 1, 0))
    df_features['pontos_visitante'] = np.where(df_features['resultado'] == 'VITORIA_VISITANTE', 3, np.where(df_features['resultado'] == 'EMPATE', 1, 0))

    times = pd.unique(df_features[['time_casa', 'time_visitante']].values.ravel('K'))
    estatisticas_times = {time: [] for time in times}

    for index, row in df_features.iterrows():
        estatisticas_times[row['time_casa']].append({'pontos': row['pontos_casa'], 'gols_marcados': row['gols_casa'], 'gols_sofridos': row['gols_visitante']})
        estatisticas_times[row['time_visitante']].append({'pontos': row['pontos_visitante'], 'gols_marcados': row['gols_visitante'], 'gols_sofridos': row['gols_casa']})

    features_calculadas = []
    JANELA = 5 

    for index, row in df_features.iterrows():
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
    df_final.dropna(inplace=True) 
    
    print(f"Engenharia de features concluída. DataFrame final com {len(df_final)} jogos e {len(df_final.columns)} colunas.")
    return df_final 
