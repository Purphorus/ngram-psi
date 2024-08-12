import json
import networkx as nx
from networkx.readwrite import json_graph
from pyvis.network import Network
from sympy import false
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer

# Função para carregar o grafo do usuário
def load_user_graph(file_path='user_graph.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    return json_graph.node_link_graph(graph_data)

# Carregar o grafo do usuário
user_graph = load_user_graph()

def visualize_graph_with_pyvis(user_graph, highlight_edges=None, highlight_nodes=None, file_name="graph_visualization.html"):
    net = Network(height="750px", width="100%", notebook=False)
    
    # Adicionando nós ao grafo Pyvis, com destaque para nós específicos
    for node in user_graph.nodes():
        if highlight_nodes and node in highlight_nodes:
            # Destaca o nó específico
            net.add_node(node, color="red")
        else:
            net.add_node(node)

    # Adicionando arestas ao grafo Pyvis, com destaque para arestas específicas
    for edge in user_graph.edges():
        color = "green" if highlight_edges and (edge in highlight_edges or (edge[1], edge[0]) in highlight_edges) else "#848484"  # Cor padrão para outras arestas
        net.add_edge(edge[0], edge[1], color=color)

    net.show(file_name, notebook=false)


def calculate_and_show_centralities(user_graph):
    # Calcula a centralidade de grau para todos os nós no grafo
    centralities = nx.degree_centrality(user_graph)
    
    # Ordena os nós pela centralidade e imprime os top 5
    sorted_centralities = sorted(centralities.items(), key=lambda x: x[1], reverse=True)
    print("Centralidades (top 5):", sorted_centralities[:5])
    
    # Extrai apenas os nós (ignorando os valores de centralidade) para os top 5
    top_nodes = [node for node, centrality in sorted_centralities[:5]]
    
    # Chama a função de visualização com os nós de maior centralidade destacados
    visualize_graph_with_pyvis(user_graph, highlight_nodes=top_nodes, file_name="centralidade_graph.html")

# Função para encontrar e visualizar o caminho baseado em um 'Dijkstra invertido'
def find_and_show_modified_dijkstra_path(user_graph, start_node, end_node):
    # Inverte a lógica dos pesos elevando-os a -1
    # Assume um peso padrão para arestas sem peso definido
    default_weight = 1  # Valor padrão para arestas sem peso
    inverted_graph = user_graph.copy()
    for u, v, data in inverted_graph.edges(data=True):
        # Usa o peso da aresta se disponível, senão usa o valor padrão
        weight = data.get('weight', default_weight)
        if weight <= 0:
            raise ValueError("Os pesos das arestas devem ser positivos e maiores que zero.")
        data['weight'] = 1 / weight
    
    # Encontra o caminho com Dijkstra usando os pesos ajustados
    path = nx.dijkstra_path(inverted_graph, source=start_node, target=end_node, weight='weight')
    print("Caminho encontrado:", path)
    
    # Converter caminho em arestas
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    visualize_graph_with_pyvis(user_graph, highlight_edges=path_edges, file_name="dijkstra_path_graph.html")

# Inicializa estruturas para rastrear as alterações feitas no grafo
nodes_added = []
edges_added = []
nodes_removed = []

def add_nodes(user_graph, nodes):
    global nodes_added  # Use a palavra-chave global se estiver modificando a variável fora da função
    for node in nodes.split(','):
        node = node.strip()
        if not user_graph.has_node(node):
            user_graph.add_node(node)
            nodes_added.append(node)  # Rastreia o nó adicionado
            print(f"Nó '{node}' adicionado.")
        else:
            print(f"Nó '{node}' já existe.")

def add_edges_with_weight(user_graph, source, targets, weight=1):
    global edges_added  # Use a palavra-chave global se estiver modificando a variável fora da função
    for target in targets.split(','):
        target = target.strip()
        if not user_graph.has_node(source) or not user_graph.has_node(target):
            print(f"Um ou ambos os nós ('{source}', '{target}') não existem.")
            continue
        user_graph.add_edge(source, target, weight=weight)
        edges_added.append((source, target, weight))  # Rastreia a aresta adicionada com peso
        print(f"Aresta entre '{source}' e '{target}' adicionada com peso {weight}.")

def remove_node(user_graph, node):
    if user_graph.has_node(node):
        user_graph.remove_node(node)
        nodes_removed.append(node)  # Rastreia o nó removido
        print(f"Nó '{node}' e suas arestas foram removidos.")
    else:
        print(f"Nó '{node}' não encontrado.")

def edit_edge_weights(user_graph, source, targets, new_weight):
    for target in targets.split(','):
        target = target.strip()
        if user_graph.has_edge(source, target):
            user_graph[source][target]['weight'] = new_weight
            print(f"Peso da aresta entre '{source}' e '{target}' atualizado para {new_weight}.")
        else:
            print(f"Aresta entre '{source}' e '{target}' não encontrada.")

def save_user_graph_to_json(user_graph, file_path='user_graph.json'):
    data = nx.readwrite.json_graph.node_link_data(user_graph)
    
    # Ajuste para incluir o peso nas arestas adicionadas
    data['metadata'] = {
        'nodes_added': nodes_added,
        'edges_added': [{'source': src, 'target': tgt, 'weight': weight} for src, tgt, weight in edges_added],
        'nodes_removed': nodes_removed
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print("Grafo salvo com sucesso.")

def calculate_node_similarity(user_graph, node1, node2):
    neighbors1 = set(user_graph.neighbors(node1))
    neighbors2 = set(user_graph.neighbors(node2))
    intersection = neighbors1.intersection(neighbors2)
    union = neighbors1.union(neighbors2)
    similarity = len(intersection) / len(union) if union else 0
    return similarity

def save_beliefs(belief_graphs, folder='Biblioteca_Crenças'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, graph in enumerate(belief_graphs):
        path = os.path.join(folder, f'crenca_{i}.json')
        data = json_graph.node_link_data(graph)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

def find_top_similar_nodes(user_graph, target_node):
    similarities = []
    for node in user_graph.nodes():
        if node != target_node:
            similarity = calculate_node_similarity(user_graph, target_node, node)
            similarities.append((node, similarity))
    # Ordenar por semelhança e pegar os top 5
    top_similar_nodes = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    return top_similar_nodes

def perform_pca_on_laplacian(user_graph, n_components=2):
    laplacian = nx.laplacian_matrix(user_graph).astype(np.float32).toarray()
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(laplacian)
    return principal_components

def plot_principal_components(principal_components):
    plt.figure(figsize=(10, 7))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA on Graph Laplacian')
    plt.show()

def perform_factor_analysis(graph, n_factors, rotation='varimax'):
    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(graph)
    
    # Initialize the factor analysis object
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    
    # Fit the factor analysis model
    fa.fit(adjacency_matrix)
    
    # Extract the loadings (factor scores)
    loadings = fa.loadings_
    return loadings

def plot_factor_loadings(loadings, graph):
    # Plot the factor loadings for each factor
    plt.figure(figsize=(12, 6))
    for i, factor_loading in enumerate(loadings.T, start=1):
        plt.subplot(1, loadings.shape[1], i)
        plt.bar(range(len(graph.nodes())), factor_loading)
        plt.title(f'Factor {i}')
        plt.xlabel('Node Index')
        plt.ylabel('Loadings')
    plt.tight_layout()
    plt.show()

    # Create a separate reference table for the nodes and their loadings
    node_labels = list(graph.nodes())
    print("Node Index - Node Label Reference Table:")
    for index, label in enumerate(node_labels):
        print(f"{index} - {label}")
    print("\nFactor Loadings Table:")
    for i, factor_loading in enumerate(loadings.T, start=1):
        print(f"Factor {i}:")
        for index, loading in enumerate(factor_loading):
            print(f"Node {node_labels[index]}: {loading:.4f}")
        print()

def find_influence_points(user_graph, show_details=False):
    betweenness = nx.betweenness_centrality(user_graph)
    influence_scores = {}
    for node in user_graph.nodes():
        temp_graph = user_graph.copy()
        temp_graph.remove_node(node)
        num_components = nx.number_connected_components(temp_graph)
        if nx.is_connected(temp_graph):
            avg_distance = nx.average_shortest_path_length(temp_graph)
        else:
            avg_distance = float('inf')  # ou alguma outra lógica para grafos desconectados
        
        # Média ponderada das métricas para score de influência
        score = (betweenness[node] * 0.5) + (1/num_components * 0.25) + (1/avg_distance if avg_distance != float('inf') else 0 * 0.25)
        influence_scores[node] = score
        
        if show_details:
            print(f"Node: {node}, Betweenness: {betweenness[node]:.4f}, Components: {num_components}, Avg Distance: {avg_distance if avg_distance != float('inf') else 'inf'}, Score: {score:.4f}")

    # Ordena e retorna os nós com maiores scores de influência
    sorted_influence = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_influence

def suggest_exploration_nodes(user_graph, interested_nodes):
    suggested_nodes = set()
    for node in interested_nodes:
        neighbors = set(user_graph.neighbors(node))
        for other_node in interested_nodes:
            if node == other_node:
                continue
            other_neighbors = set(user_graph.neighbors(other_node))
            # Encontra nós que são vizinhos de ambos, mas não estão na lista de nós interessados
            common_neighbors = (neighbors & other_neighbors) - set(interested_nodes)
            suggested_nodes.update(common_neighbors)

    if suggested_nodes:
        print("Nós sugeridos para exploração:", suggested_nodes)
    else:
        print("Não foram encontrados nós sugeridos para exploração baseados nos nós de interesse fornecidos.")

def find_bridges_in_graph(user_graph):
    bridges = list(nx.bridges(user_graph))
    if bridges:
        print("Pontes encontradas no grafo:")
        for bridge in bridges:
            print(bridge)
    else:
        print("Nenhuma ponte encontrada no grafo.")

# Interface de usuário para escolha da análise
while True:
    choice = input("Escolha uma opção:\n"
                   "1. Centralidade\n"
                   "2. Dijkstra Modificado\n"
                   "3. Adicionar nós\n"
                   "4. Conectar nós\n"
                   "5. Editar pesos de arestas\n"
                   "6. Apagar nó\n"
                   "7. Análise PCA\n"
                   "8. Similaridade de nó\n"
                   "9. Análise Fatorial\n"
                   "10. Pontos de influencia (ver doc)\n"
                   "11. Nós de interesse\n"
                   "12. Revelar pontes\n"
                   "0. Sair\n> ")
    if choice == '1':
        calculate_and_show_centralities(user_graph)
    elif choice == '2':
        start_node = input("Nó de início: ")
        end_node = input("Nó de fim: ")
        find_and_show_modified_dijkstra_path(user_graph, start_node, end_node)
    elif choice == '3':
        nodes = input("Nomes dos nós para adicionar (separados por vírgula): ")
        add_nodes(user_graph, nodes)
    elif choice == '4':
        source = input("Nó de origem: ")
        targets = input("Nós de destino (separados por vírgula): ")
        add_edges_with_weight(user_graph, source, targets)
    elif choice == '5':
        source = input("Nó de origem: ")
        targets = input("Nós de destino (separados por vírgula): ")
        new_weight = int(input("Novo peso: "))
        edit_edge_weights(user_graph, source, targets, new_weight)
    elif choice == '6':
        node = input("Nome do nó para apagar: ")
        remove_node(user_graph, node)
    if choice == '7':
        principal_components = perform_pca_on_laplacian(user_graph)
        plot_principal_components(principal_components)
    elif choice == '8':
        target_node = input("Informe o nó de interesse: ")
        if user_graph.has_node(target_node):
            top_similar_nodes = find_top_similar_nodes(user_graph, target_node)
            print("Top 5 nós semelhantes:")
            for node, similarity in top_similar_nodes:
                print(f"{node}: {similarity*100:.2f}% semelhante")
        else:
            print("Nó não encontrado no grafo.")  
    elif choice == '9':
        n_factors_input = input("Informe o número de fatores para análise: ")
        try:
            n_factors = int(n_factors_input)  # Tenta converter a entrada para inteiro
            loadings = perform_factor_analysis(user_graph, n_factors=n_factors)
            plot_factor_loadings(loadings, user_graph)
        except ValueError:
            print("Por favor, insira um número inteiro válido para o número de fatores.")
    elif choice == '10':  # Supondo que '10' seja o próximo número disponível na sua lista de opções
        show_details = input("Deseja ver os detalhes das métricas para cada nó? (s/n): ").lower() == 's'
        top_influence_points = find_influence_points(user_graph, show_details)
        print("Top pontos de influência (nó, score):")
        for node, score in top_influence_points[:5]:  # Limitando a exibição aos top 5
            print(f"{node}: {score:.4f}")
    elif choice == '11':  # Ajuste conforme necessário para sua estrutura de menu
        input_nodes = input("Insira os nós de interesse, separados por vírgula: ")
        interested_nodes = [node.strip() for node in input_nodes.split(',')]
        suggest_exploration_nodes(user_graph, interested_nodes)        
    elif choice == '12':  
        find_bridges_in_graph(user_graph)        
    elif choice == '0':
        save_choice = input("Deseja salvar as alterações no grafo? (s/n): ")
        if save_choice.lower() == 's':
            save_user_graph_to_json(user_graph, 'user_graph.json')
        print("Saindo...")
        break
    else:
        print("Opção inválida.")