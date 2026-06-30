import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE

print("Iniciando pipeline(versão para ambientes restritos)...")
try:
    df = pd.read_csv('data.csv', sep=';')
except FileNotFoundError:
    df = pd.read_csv('data.csv')

colunas_estudante = [
    'Age at enrollment', 'Gender', 'Scholarship holder', 'Displaced',
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)'
]

coluna_curso = 'Course'
coluna_devedor = 'Debtor'

target_map = {'Graduate': 0, 'Enrolled': 1, 'Dropout': 2}
df['target_num'] = df['Target'].map(target_map)
df = df.dropna(subset=colunas_estudante + [coluna_curso, coluna_devedor, 'target_num'])

num_estudantes = len(df)
print(f"Dataset carregado. Total de registros: {num_estudantes}")

print("Simulando Message Passing e Atenção Relacional")
X_tab = df[colunas_estudante].values
vec_devedor = df[coluna_devedor].values.reshape(-1, 1)

np.random.seed(42)
efeito_curso = np.sin(df[coluna_curso].values).reshape(-1, 1)

W_proj = np.random.randn(X_tab.shape[1], 30) * 0.1
Z_estudante_base = np.dot(X_tab, W_proj)

Z_simulado = np.hstack([Z_estudante_base, vec_devedor * 1.5, efeito_curso * 0.5])

print(f"Matriz de Embeddings simulada com sucesso: {Z_simulado.shape}")

labels_reais = df['target_num'].values

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(Z_simulado, labels_reais)

print(f"Distribuição original do Target: {np.bincount(labels_reais)}")
print(f"Distribuição após aplicação do SMOTE: {np.bincount(y_balanced)}")

classificador_xgb = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=5, 
    learning_rate=0.05, 
    eval_metric='mlogloss'
)
classificador_xgb.fit(X_balanced, y_balanced)

probs = classificador_xgb.predict_proba(Z_simulado[0].reshape(1, -1))[0]

print("\n=== DIAGNÓSTICO DE RISCO HEAL-GNN (SIMULADO - ALUNO ÍNDICE 0) ===")
print(f"Probabilidade de Sucesso (Graduate): {probs[0]*100:.2f}%")
print(f"Probabilidade de Retenção (Enrolled): {probs[1]*100:.2f}%")
print(f"Alerta de Risco de Evasão (Dropout): {probs[2]*100:.2f}%")