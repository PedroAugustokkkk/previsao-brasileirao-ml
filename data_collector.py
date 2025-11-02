import requests     
import pandas as pd 
import json          
import os            
from dotenv import load_dotenv 


load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")

def fetch_brasileirao_data(season: int, status: str = None):
    """
    Busca os dados de uma temporada do Brasileirão na API-Football.

    Args:
        season (int): O ano da temporada a ser buscada (ex: 2025).

    Returns:
        pd.DataFrame: Um DataFrame com os dados brutos dos jogos, ou None se falhar.
    """
    
    if not API_KEY:
        print("Erro: A variável de ambiente FOOTBALL_API_KEY não foi encontrada.")
        print("Verifique se você criou o arquivo .env e definiu a chave corretamente.")
        return None 
        
    url = "https://v3.football.api-sports.io/fixtures"
    
    parametros = {"league": "71", "season": str(season)}
    if status:
        parametros["status"] = status
        
    headers = {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key": API_KEY 
    }

    try:
        response = requests.get(url, headers=headers, params=parametros)
        response.raise_for_status()
        
        dados_json = response.json()
        
        lista_jogos = []
        for jogo in dados_json.get('response', []):
            lista_jogos.append({
                'data': jogo['fixture']['date'],
                'time_casa': jogo['teams']['home']['name'],
                'time_visitante': jogo['teams']['away']['name'],
                'gols_casa': jogo['goals']['home'],
                'gols_visitante': jogo['goals']['away']
            })
            
        df_jogos = pd.DataFrame(lista_jogos)
        print(f"Dados de {len(df_jogos)} jogos da temporada {season} coletados com sucesso.")
        return df_jogos 

    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar dados da API: {e}")
        return None 