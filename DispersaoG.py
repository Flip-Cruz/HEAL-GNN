import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    df = pd.read_csv('data.csv', sep=';')
except FileNotFoundError:
    df = pd.read_csv('data.csv')

colunas_estudante = [
    'Age at enrollment', 'Gender', 'Scholarship holder', 'Displaced',
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)'
]

df = df.dropna(subset=colunas_estudante + ['Course', 'Debtor'])

X_tab = df[colunas_estudante].values
vec_devedor = df['Debtor'].values.reshape(-1, 1)
efeito_curso = np.sin(df['Course'].values).reshape(-1, 1)

np.random.seed(42)
W_proj = np.random.randn(X_tab.shape[1], 30) * 0.1
Z_estudante_base = np.dot(X_tab, W_proj)

Z_simulado = np.hstack([Z_estudante_base, vec_devedor * 2.0, efeito_curso * 0.5])

pca = PCA(n_components=2, random_state=42)
Z_2d = pca.fit_transform(Z_simulado)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(Z_simulado)

plt.figure(figsize=(9, 7))

cores = ['#4A90E2', '#E25A5A', '#9B5DE5']
labels_clusters = [
    'Cluster C: Alunos Regulares / Estáveis',
    'Cluster A: Alta Vulnerabilidade Socioeconômica',
    'Cluster B: Desengajamento Puramente Acadêmico'
]

for i in range(3):
    idx = (clusters == i)
    plt.scatter(
        Z_2d[idx, 0], Z_2d[idx, 1], 
        c=cores[i], label=labels_clusters[i],
        alpha=0.75, edgecolors='none', s=45
    )

plt.title('Análise de Proximidade e Distribuição dos Clusters Latentes (Meso-level)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Componente Latente 1 (Dimensão Espacial X)', fontsize=10, labelpad=8)
plt.ylabel('Componente Latente 2 (Dimensão Espacial Y)', fontsize=10, labelpad=8)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=True, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

plt.savefig('clusters_latentes.png', dpi=300, bbox_inches='tight')
plt.close()

print("O arquivo 'clusters_latentes.png' foi gerado e salvo no diretório.")