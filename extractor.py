import os
import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from networkx.linalg.laplacianmatrix import laplacian_matrix

def load_user_graph(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)

def optimal_cluster_number(user_graph, max_clusters=10):
    # Calculando a matriz laplaciana e garantindo que está em formato de array esparsa
    laplacian = laplacian_matrix(user_graph).astype(np.float32)
    # Convertendo de array esparsa para array densa usando .toarray() ao invés de .todense()
    laplacian_dense = laplacian.toarray()
    
    best_score = -1
    best_cluster = 2

    for num_clusters in range(2, max_clusters+1):
        # Calculando autovalores e autovetores
        eigenvals, eigenvects = np.linalg.eigh(laplacian_dense)
        # Utilizando os autovetores para clustering
        vects = eigenvects[:, 1:num_clusters]
        
        # Aplicando KMeans nos autovetores
        kmeans = KMeans(n_clusters=num_clusters).fit(vects)
        labels = kmeans.labels_
        
        # Calculando a pontuação de silhueta para avaliar a configuração do cluster
        score = silhouette_score(vects, labels)
        
        # Atualizando o melhor número de clusters se a pontuação de silhueta for melhor
        if score > best_score:
            best_score = score
            best_cluster = num_clusters
    
    return best_cluster

def perform_spectral_analysis(user_graph, num_clusters=2):
    # Calcula a matriz laplaciana
    laplacian = nx.laplacian_matrix(user_graph).astype(np.float32)
    # Assegura que a matriz laplaciana seja convertida para um array do Numpy
    laplacian_array = np.asarray(laplacian.todense())
    
    # Calcula autovalores e autovetores a partir do array
    eigenvals, eigenvects = np.linalg.eigh(laplacian_array)
    
    # Utiliza os autovetores correspondentes aos menores autovalores para clustering
    vects = eigenvects[:, 1:num_clusters]
    
    # Converte vects para um array do Numpy antes de usar no KMeans
    vects_array = np.asarray(vects)
    
    # Aplica k-means nos autovetores para identificar clusters
    kmeans = KMeans(n_clusters=num_clusters).fit(vects_array)
    labels = kmeans.labels_
    
    # Cria um mapeamento de nó para cluster/label
    node_cluster_mapping = {list(user_graph.nodes())[i]: label for i, label in enumerate(labels)}
    
    return node_cluster_mapping


def save_beliefs(belief_graphs, folder='Biblioteca_Crenças'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, graph in enumerate(belief_graphs):
        path = os.path.join(folder, f'crenca_{i}.json')
        data = json_graph.node_link_data(graph)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

def extract_and_save_beliefs(user_graph_path):
    user_graph = load_user_graph(user_graph_path)
    
    choice = input("Escolha o número de clusters ou 'auto' para determinação automática: ")
    if choice.lower() == 'auto':
        num_clusters = optimal_cluster_number(user_graph)
    else:
        num_clusters = int(choice)
    
    node_cluster_mapping = perform_spectral_analysis(user_graph, num_clusters=num_clusters)
    user_graph = load_user_graph(user_graph_path)
    
    # Perform spectral analysis to get node to cluster mapping
    node_cluster_mapping = perform_spectral_analysis(user_graph)
    
    # Determinar o número de clusters baseado no mapeamento
    num_clusters = len(set(node_cluster_mapping.values()))
    
    belief_graphs = []
    for cluster_id in range(num_clusters):
        # Filtrar nós por cluster_id
        nodes_in_cluster = [node for node, cluster in node_cluster_mapping.items() if cluster == cluster_id]
        
        # Criar subgrafo para o cluster atual
        subgraph = user_graph.subgraph(nodes_in_cluster)
        belief_graphs.append(subgraph)
    
    # Salvar cada subgrafo/crença
    save_beliefs(belief_graphs)

# Exemplo de uso
user_graph_path = '' #lembre de colocar o path do grafo .json gerado
extract_and_save_beliefs(user_graph_path)
