# Arquivo: data_collector.py

# Importações necessárias para esta tarefa.
import requests
import pandas as pd
import json

# A chave da API é uma constante que pode ser facilmente alterada aqui.
API_KEY = "SUA_API_KEY_AQUI"

def fetch_brasileirao_data(season: int):
    """
    Busca os dados de uma temporada do Brasileirão na API-Football.

    Args:
        season (int): O ano da temporada a ser buscada (ex: 2025).

    Returns:
        pd.DataFrame: Um DataFrame com os dados brutos dos jogos, ou None se falhar.
    """
    # Endpoint da API para buscar as partidas (fixtures).
    url = "https://v3.football.api-sports.io/fixtures"
    
    # Parâmetros da busca, usando o ano que foi passado para a função.
    parametros = {"league": "71", "season": str(season)}
    
    # Cabeçalhos para autenticação.
    headers = {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key": API_KEY
    }

    try:
        # Faz a requisição para a API.
        response = requests.get(url, headers=headers, params=parametros)
        # Levanta um erro caso a resposta não seja de sucesso (ex: erro 404, 500).
        response.raise_for_status()
        
        # Converte a resposta JSON em um objeto Python.
        dados_json = response.json()
        
        # Processa o JSON para criar uma lista de jogos.
        lista_jogos = []
        for jogo in dados_json.get('response', []):
            lista_jogos.append({
                'data': jogo['fixture']['date'],
                'time_casa': jogo['teams']['home']['name'],
                'time_visitante': jogo['teams']['away']['name'],
                'gols_casa': jogo['goals']['home'],
                'gols_visitante': jogo['goals']['away']
            })
            
        # Converte a lista em um DataFrame do pandas.
        df_jogos = pd.DataFrame(lista_jogos)
        print(f"Dados de {len(df_jogos)} jogos da temporada {season} coletados com sucesso.")
        return df_jogos # Retorna o DataFrame criado.

    except requests.exceptions.RequestException as e:
        # Em caso de erro de conexão ou na API, informa o usuário.
        print(f"Erro ao buscar dados da API: {e}")
        return None # Retorna None para indicar que a coleta falhou.
