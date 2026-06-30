import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def load_and_process_data(filepath):
    """
    Carrega o dataset e prepara as variáveis para a construção do grafo.
    """
    
    df = pd.read_csv(filepath, sep=';')
    
    df_sample = df.sample(n=150, random_state=42).reset_index()
    
    return df_sample

def build_heterogeneous_graph(df):
    """
    Constrói um grafo NetworkX mapeando estudantes, cursos e status financeiro.
    """
    G = nx.Graph()
    
    for idx, row in df.iterrows():
        student_node = f"S_{idx}"
        course_node = f"Course_{row['Course']}"
        
        is_debtor = row['Debtor'] == 1
        is_delayed = row['Tuition fees up to date'] == 0
        financial_node = "Fin_At_Risk" if (is_debtor or is_delayed) else "Fin_OK"
        
        G.add_node(student_node, 
                   type='Student', 
                   target=row['Target'],
                   age=row['Age at enrollment'],
                   grade_1st=row['Curricular units 1st sem (grade)'],
                   grade_2nd=row['Curricular units 2nd sem (grade)'])
        
        G.add_node(course_node, type='Course')
        
        G.add_node(financial_node, type='FinancialStatus')
        
        G.add_edge(student_node, course_node, relation='enrolled_in')
        
        G.add_edge(student_node, financial_node, relation='has_status')
        
        if row['Curricular units 2nd sem (grade)'] < row['Curricular units 1st sem (grade)']:
            G.nodes[student_node]['trajectory'] = 'declining'
        else:
            G.nodes[student_node]['trajectory'] = 'stable_or_improving'

    return G

def visualize_graph(G):
    """
    Plota o grafo heterogêneo utilizando cores para diferenciar os nós e o risco de evasão.
    """
    plt.figure(figsize=(14, 10))
    
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    
    color_map = []
    size_map = []
    
    for node, data in G.nodes(data=True):
        if data['type'] == 'Course':
            color_map.append('lightblue')
            size_map.append(800)
        elif data['type'] == 'FinancialStatus':
            color_map.append('lightcoral' if node == 'Fin_At_Risk' else 'lightgreen')
            size_map.append(1000)
        elif data['type'] == 'Student':
            size_map.append(150)
            # Colorir estudantes com base no Desfecho (Target)
            if data['target'] == 'Dropout':
                color_map.append('red')
            elif data['target'] == 'Graduate':
                color_map.append('green')
            else: # Enrolled
                color_map.append('orange')
                
    edge_colors = ['grey' if G[u][v]['relation'] == 'enrolled_in' else 'black' for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.5, width=1.0)
    
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=size_map, alpha=0.9, edgecolors='white')
    
    labels = {n: n for n, d in G.nodes(data=True) if d['type'] in ['Course', 'FinancialStatus']}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')
    
    dropout_patch = mpatches.Patch(color='red', label='Student: Dropout')
    graduate_patch = mpatches.Patch(color='green', label='Student: Graduate')
    enrolled_patch = mpatches.Patch(color='orange', label='Student: Enrolled')
    course_patch = mpatches.Patch(color='lightblue', label='Course Node')
    fin_patch = mpatches.Patch(color='lightcoral', label='Financial Node')
    
    plt.legend(handles=[dropout_patch, graduate_patch, enrolled_patch, course_patch, fin_patch], 
               loc='upper right', fontsize=10)
    
    plt.title("HEAL-GNN: Topologia de Interações entre Estudantes, Cursos e Finanças", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def gnn_ensemble_pipeline_mock(G, df):
    """
    Estrutura conceitual de como a saída do grafo alimentaria o modelo preditivo.
    (Requer PyTorch Geometric e XGBoost em produção)
    """
    print("\n--- Pipeline Preditivo HEAL-GNN (Scaffold) ---")
    print("1. [GNN Layer]: Extraindo embeddings estruturais dos nós...")
    print("2. [Concatenação]: Unindo atributos estáticos (idade, notas) com os embeddings do grafo...")
    print("3. [SMOTE]: Balanceando as classes (Dropout vs Graduate)...")
    print("4. [XGBoost/LightGBM]: Treinando ensemble com árvores de decisão baseadas em gradiente...")
    print("5. [Output]: Probabilidade de risco de evasão calculada por estudante.")

if __name__ == "__main__":

    filepath = 'data.csv'
    
    try:
        df = load_and_process_data(filepath)
        print(f"Dataset carregado com sucesso. Amostra de {len(df)} estudantes para visualização.")
        
        G = build_heterogeneous_graph(df)
        print(f"Grafo construído: {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")
        
        visualize_graph(G)
        gnn_ensemble_pipeline_mock(G, df)
        
    except FileNotFoundError:
        print("Erro: O arquivo 'data.csv' não foi encontrado. Verifique o caminho.")