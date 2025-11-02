# -*- coding: utf-8 -*-
# Comentários em cada linha, como solicitado.

# Importa as bibliotecas necessárias para esta tarefa.
import requests       # Para fazer as chamadas de API (HTTP requests).
import pandas as pd   # Para organizar os dados em tabelas (DataFrames).
import json           # Para lidar com a resposta da API (formato JSON).
import os             # Importa o módulo 'os' para acessar variáveis de ambiente.
from dotenv import load_dotenv # Importa a função específica para carregar o .env.

# --- Carregando as Variáveis de Ambiente ---
# A função load_dotenv() procura por um arquivo .env no diretório
# e carrega as variáveis definidas nele para o ambiente do sistema.
load_dotenv()

# --- Configuração da Requisição ---
# Agora, em vez de escrever a chave aqui, nós a lemos do ambiente.
# os.getenv() busca a variável de ambiente com o nome que passamos.
API_KEY = os.getenv("FOOTBALL_API_KEY") # Pega o valor da variável FOOTBALL_API_KEY do arquivo .env.

# A função agora aceita um 'status' (ex: "FT" para finalizado, "NS" para não iniciado)
def fetch_brasileirao_data(season: int, status: str = None):
    """
    Busca os dados de uma temporada do Brasileirão na API-Football.

    Args:
        season (int): O ano da temporada a ser buscada (ex: 2025).
        status (str, optional): O status do jogo (ex: "FT", "NS").

    Returns:
        pd.DataFrame: Um DataFrame com os dados brutos dos jogos, ou None se falhar.
    """
    
    # Verifica se a chave de API foi carregada corretamente.
    if not API_KEY:
        # Se a chave não foi encontrada, avisa o usuário e interrompe.
        print("Erro: A variável de ambiente FOOTBALL_API_KEY não foi encontrada.") # Mensagem de erro
        print("Verifique se você criou o arquivo .env e definiu a chave corretamente.") # Dica de correção
        return None # Retorna None para indicar a falha.
        
    # Endpoint da API para buscar as partidas (fixtures).
    url = "https://v3.football.api-sports.io/fixtures"
    
    # Parâmetros da busca, usando o ano que foi passado para a função.
    parametros = {"league": "71", "season": str(season)}
    
    # Adiciona o status aos parâmetros SE ele for fornecido
    if status:
        parametros["status"] = status # Isso filtrará os jogos na API
    
    # Cabeçalhos para autenticação, usando a chave que carregamos do .env.
    headers = {
        "x-rapidapi-host": "v3.football.api-sports.io", # O "host" do servidor da API
        "x-rapidapi-key": API_KEY # A sua chave secreta para autenticação
    }

    try:
        # Faz a requisição para a API.
        print(f"Buscando dados da API para temporada={season}, status={status}...") # Log
        response = requests.get(url, headers=headers, params=parametros)
        # Levanta um erro caso a resposta não seja de sucesso (ex: erro 404, 500).
        response.raise_for_status()
        
        # Converte a resposta JSON em um objeto Python.
        dados_json = response.json()
        
        # Processa o JSON para criar uma lista de jogos.
        lista_jogos = []
        # Itera sobre cada jogo encontrado na chave 'response' do JSON
        for jogo in dados_json.get('response', []):
            # Adiciona um dicionário formatado à nossa lista
            lista_jogos.append({
                'data': jogo['fixture']['date'], # Data e hora do jogo
                'time_casa': jogo['teams']['home']['name'], # Nome do time da casa
                'time_visitante': jogo['teams']['away']['name'], # Nome do time visitante
                'gols_casa': jogo['goals']['home'], # Gols do time da casa
                'gols_visitante': jogo['goals']['away'] # Gols do time visitante
            })
            
        # Converte a lista em um DataFrame do pandas.
        df_jogos = pd.DataFrame(lista_jogos)
        print(f"Dados de {len(df_jogos)} jogos coletados com sucesso.") # Log de sucesso
        return df_jogos # Retorna o DataFrame criado.

    except requests.exceptions.RequestException as e:
        # Em caso de erro de conexão ou na API, informa o usuário.
        print(f"Erro ao buscar dados da API: {e}")
        return None # Retorna None para indicar que a coleta falhou.
