import requests  # Importa a biblioteca para fazer requisições HTTP (acessar a API).
import pandas as pd  # Importa a biblioteca pandas para organizar os dados em uma tabela (DataFrame).
import json # Importa a biblioteca para trabalhar com o formato de dados JSON, que é o que a API retorna.
from dotenv import load_dotenv # para gerenciar variáveis de ambiente
load_dotenv()
# O endpoint 'fixtures' é o que nos dá os dados das partidas de futebol.
url = "https://v3.football.api-sports.io/fixtures"

# 'league': '71' é o ID da Série A do Brasileirão na API-Football.
# 'season': '2024' é o ano da temporada que queremos consultar.
parametros = {
    "league": "71",
    "season": "2024"
}

# Cabeçalhos da requisição: aqui você se autentica na API.

headers = {
    "x-rapidapi-host": "v3.football.api-sports.io",
    "x-rapidapi-key": "API_KEY"
}

# A função 'requests.get' envia a requisição para o servidor da API.
# passamos a url, os parâmetros de busca e os cabeçalhos de autenticação.
response = requests.get(url, headers=headers, params=parametros)

# A API retorna os dados em formato JSON. Usamos 'json.loads' para transformar o texto da resposta em um objeto Python.
dados_json = json.loads(response.text)

# Vamos criar uma lista para armazenar os dados de cada jogo de forma organizada.
lista_jogos = []

# O JSON da API tem uma estrutura de aninhamento. Navegamos até a lista de 'response' que contém os jogos.
for jogo in dados_json['response']:
    # Para cada jogo na lista, extraímos as informações que nos interessam.
    data_jogo = jogo['fixture']['date'] # data e hora do jogo.
    time_casa = jogo['teams']['home']['name'] # nome do time da casa.
    time_visitante = jogo['teams']['away']['name'] # nome do time visitante.
    gols_casa = jogo['goals']['home'] # gols marcados pelo time da casa.
    gols_visitante = jogo['goals']['away'] # gols marcados pelo time visitante.
    
    # Adicionamos as informações extraídas como um dicionário à nossa lista.
    lista_jogos.append({
        'data': data_jogo,
        'time_casa': time_casa,
        'time_visitante': time_visitante,
        'gols_casa': gols_casa,
        'gols_visitante': gols_visitante
    })

# Convertendo a lista de dicionários em um DataFrame do pandas para facilitar a manipulação.
df_jogos = pd.DataFrame(lista_jogos)

print("Dados dos jogos do Brasileirão 2024:") # Imprime um título.

print(df_jogos.head()) # Exibe as 5 primeiras linhas do nosso DataFrame para verificação.
