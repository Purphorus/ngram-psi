import json
import networkx as nx
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from string import punctuation

# Certifique-se de que o pacote 'punkt' ou 'floresta' da NLTK esteja baixado
nltk.download('floresta')
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('wordnet')

nlp = spacy.load("pt_core_news_sm")

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def load_synonyms(file_path=''): #coloque aqui o path absoluto do seu sinonimos.txt
    synonyms = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(':')
            if len(parts) == 2:
                words, synonym = parts
                for word in words.split(','):
                    synonyms[word.strip()] = synonym
    return synonyms

def load_custom_stopwords(file_path=''):#coloque aqui o path absoluto do seu stopwords.txt
    with open(file_path, 'r', encoding='utf-8') as file:
        custom_stopwords = file.read().splitlines()
    return set(custom_stopwords)

def preprocess_text(text, stopwords_file='', synonyms_file=''): #lembre de colocar os paths novamente
    # Carregar stopwords personalizadas do arquivo
    custom_stopwords = load_custom_stopwords(stopwords_file)
    
    # Carregar sinônimos do arquivo
    synonyms = load_synonyms(synonyms_file)
    
    # Define o conjunto padrão de stopwords e adiciona sinais de pontuação
    standard_stopwords = set(stopwords.words('portuguese')) | set(punctuation)
    
    # Combina as stopwords padrão com as personalizadas
    all_stopwords = standard_stopwords | custom_stopwords
    
    # Tokenização e conversão para minúsculas
    tokens = word_tokenize(text.lower())
    
    # Substituir sinônimos, se fornecidos
    tokens = [synonyms.get(token, token) for token in tokens]
    
    # Remoção de stopwords
    filtered_tokens = [token for token in tokens if token not in all_stopwords]
    
    # Lematização com spaCy
    lemmatized_tokens = [token.lemma_ for token in nlp(' '.join(filtered_tokens))]
    
    # Stemming com RSLPStemmer para tokens lematizados
    stemmer = RSLPStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    
    #return lemmatized_tokens#
    return filtered_tokens

# Mecanismo ao contrário com steamm antes de leamma
#def preprocess_text(text):
    # Tokenização
    tokens = word_tokenize(text.lower())  # Converte para minúsculas durante a tokenização
    
    # Remoção de stopwords
    stop_words = set(stopwords.words('portuguese'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = RSLPStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    # Lematização em português
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    
    return lemmatized_tokens#

def create_entry_graph_from_file(file_path, n=2):
    text = read_text_file(file_path)
    preprocessed_tokens = preprocess_text(text)
    
    G = nx.Graph()
    n_grams = list(ngrams(preprocessed_tokens, n))

    for gram in n_grams:
        if not G.has_edge(gram[0], gram[1]):
            G.add_edge(gram[0], gram[1], weight=1)
        else:
            G[gram[0]][gram[1]]['weight'] += 1
    return G

def update_user_graph(user_graph, entry_graph):
    for u, v, data in entry_graph.edges(data=True):
        if user_graph.has_edge(u, v):
            user_graph[u][v]['weight'] += data['weight']
        else:
            user_graph.add_edge(u, v, weight=data['weight'])
    return user_graph

def save_user_graph(user_graph, text_files, file_path='user_graph.json'):
    graph_data = nx.readwrite.json_graph.node_link_data(user_graph)
    graph_data['texts'] = text_files
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4)

def load_user_graph(file_path='user_graph.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    user_graph = nx.readwrite.json_graph.node_link_graph(graph_data)
    text_files = graph_data.get('texts', [])
    return user_graph, text_files

# Inicialização ou carregamento do Grafo de Usuário Geral
try:
    user_graph, used_texts = load_user_graph()
    print("Grafo carregado com sucesso.")
except FileNotFoundError:
    user_graph = nx.Graph()
    used_texts = []
    print("Iniciando um novo grafo de usuário geral.")

# Solicita ao usuário para inserir os caminhos dos arquivos de texto
input_files = input("Insira os caminhos dos arquivos de texto, separados por vírgula: ")
file_paths = input_files.split(',')

for file_path in file_paths:
    file_path = file_path.strip()  # Remove espaços em branco extra
    if file_path and file_path not in used_texts:  # Verifica se o arquivo já foi processado
        try:
            entry_graph = create_entry_graph_from_file(file_path)
            user_graph = update_user_graph(user_graph, entry_graph)
            used_texts.append(file_path)  # Atualiza a lista de textos processados
            print(f"Grafo atualizado com {file_path}.")
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {file_path}")
    elif file_path in used_texts:
        print(f"O arquivo {file_path} já foi processado anteriormente.")
    else:
        print("Nenhum caminho de arquivo válido fornecido.")

# Salva o Grafo de Usuário Geral atualizado após processar todos os novos arquivos
if file_paths:  # Se pelo menos um caminho de arquivo foi fornecido
    save_user_graph(user_graph, used_texts)
    print("Grafo de Usuário Geral atualizado e salvo com sucesso.")
