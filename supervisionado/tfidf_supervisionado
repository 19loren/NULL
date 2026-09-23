import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import hstack, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

# descobre a pasta exata onde este script esta salvo
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))

# cria as pastas 'graficos' e 'tf-idf' 
pasta_graficos = os.path.join(DIRETORIO_ATUAL, "graficos")
pasta_dados = os.path.join(DIRETORIO_ATUAL, "tf-idf")

os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_dados, exist_ok=True)

print(f"Pastas '{pasta_graficos}' e '{pasta_dados}' prontas.")

# preparacao dos dados e vetorizacao
caminho_excel = os.path.join(DIRETORIO_ATUAL, 'healthver_preparado_completo_3classes.xlsx')
df = pd.read_excel(caminho_excel)

# garantir tratamento de textos nulos e converter para string
df['question'] = df['question'].fillna('').astype(str)
df['claim']    = df['claim'].fillna('').astype(str)
df['evidence'] = df['evidence'].fillna('').astype(str)

# divisao treino, validaçao e teste ANTES da vetorizaçao
df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label_multiclasse'])
df_val, df_test   = train_test_split(df_temp, test_size=0.50, random_state=42, stratify=df_temp['label_multiclasse'])

# vetorizadores otimizados
vec_question = TfidfVectorizer(max_features=400,  min_df=2, stop_words='english', ngram_range=(1, 2))
vec_claim    = TfidfVectorizer(max_features=800,  min_df=2, stop_words='english', ngram_range=(1, 2))
vec_evidence = TfidfVectorizer(max_features=1200, min_df=2, stop_words='english', ngram_range=(1, 2))

# treino
X_tr_q = vec_question.fit_transform(df_train['question'])
X_tr_c = vec_claim.fit_transform(df_train['claim'])
X_tr_e = vec_evidence.fit_transform(df_train['evidence'])
X_train_stacked = hstack([X_tr_q, X_tr_c, X_tr_e]).tocsr()

# validaçao
X_val_q = vec_question.transform(df_val['question'])
X_val_c = vec_claim.transform(df_val['claim'])
X_val_e = vec_evidence.transform(df_val['evidence'])
X_val_stacked = hstack([X_val_q, X_val_c, X_val_e]).tocsr()

# teste
X_test_q = vec_question.transform(df_test['question'])
X_test_c = vec_claim.transform(df_test['claim'])
X_test_e = vec_evidence.transform(df_test['evidence'])
X_test_stacked = hstack([X_test_q, X_test_c, X_test_e]).tocsr()

# rotulos (y)
y_train = df_train['label_multiclasse'].values
y_val   = df_val['label_multiclasse'].values
y_test  = df_test['label_multiclasse'].values

# salvar as matrizes otimizadas DENTRO da pasta tf-idf
save_npz(os.path.join(pasta_dados, 'X_train_tfidf.npz'), X_train_stacked)
save_npz(os.path.join(pasta_dados, 'X_val_tfidf.npz'), X_val_stacked)
save_npz(os.path.join(pasta_dados, 'X_test_tfidf.npz'), X_test_stacked)

np.save(os.path.join(pasta_dados, 'y_train.npy'), y_train)
np.save(os.path.join(pasta_dados, 'y_val.npy'), y_val)
np.save(os.path.join(pasta_dados, 'y_test.npy'), y_test)

print(f"Matrizes TF-IDF salvas em '{pasta_dados}'. Formato: {X_train_stacked.shape}")

# config visuais
AZUL = "#183A8F"
TURQUESA = "#4EB8C5"
TURQUESA_CLARO = "#7FD0D8"
FUNDO_CLARO = "#F1FAFB"

cmap_grupo = LinearSegmentedColormap.from_list(
    "grupo_null",
    [FUNDO_CLARO, TURQUESA_CLARO, TURQUESA, AZUL]
)

# carregamento de dados p treino
X_train = load_npz(os.path.join(pasta_dados, 'X_train_tfidf.npz'))
X_val   = load_npz(os.path.join(pasta_dados, 'X_val_tfidf.npz'))
X_test  = load_npz(os.path.join(pasta_dados, 'X_test_tfidf.npz'))

y_train_raw = np.load(os.path.join(pasta_dados, 'y_train.npy'), allow_pickle=True)
y_val_raw   = np.load(os.path.join(pasta_dados, 'y_val.npy'), allow_pickle=True)
y_test_raw  = np.load(os.path.join(pasta_dados, 'y_test.npy'), allow_pickle=True)

# padronizaçao dos rotulos numericos (necessario para o XGBoost)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train_raw)
y_val   = encoder.transform(y_val_raw)
y_test  = encoder.transform(y_test_raw)

classes_nomes = encoder.classes_

# limpa o nome do modelo para usar no nome do arquivo salvo
def nome_arquivo_limpo(nome):
    return nome.replace(" ", "_").replace("-", "").replace("ç", "c").replace("ã", "a").strip()

# funcoes de avaliacao e visualizacao
def calcular_metricas(y_real, y_pred):
    return {
        "Acurácia": accuracy_score(y_real, y_pred),
        "Precisão macro": precision_score(y_real, y_pred, average="macro", zero_division=0),
        "Recall macro": recall_score(y_real, y_pred, average="macro", zero_division=0),
        "F1 macro": f1_score(y_real, y_pred, average="macro", zero_division=0)
    }

def grafico_metricas(nome, metricas):
    nomes = ["Acurácia", "Precisão macro", "Recall macro", "F1 macro"]
    valores = [metricas[m] for m in nomes]

    plt.figure(figsize=(7, 4))
    barras = plt.bar(nomes, valores, color=[AZUL, TURQUESA, TURQUESA_CLARO, AZUL])
    plt.ylim(0, 1)
    plt.ylabel("Valor")
    plt.title(f"{nome} - métricas na validação")

    for barra, valor in zip(barras, valores):
        plt.text(barra.get_x() + barra.get_width() / 2, valor + 0.02, f"{valor:.3f}", ha="center")

    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_graficos, f"metricas_{nome_arquivo_limpo(nome)}.png"))
    plt.close()

def grafico_matriz_confusao(nome, y_real, y_pred):
    cm = confusion_matrix(y_real, y_pred)
    classes = list(classes_nomes)

    plt.figure(figsize=(6.5, 5.2))
    plt.imshow(cm, cmap=cmap_grupo)
    plt.title(f"matriz de confusão - {nome}", color=AZUL)
    plt.xlabel("classe prevista", color=AZUL)
    plt.ylabel("classe real", color=AZUL)
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)

    max_val = cm.max()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            valor = cm[i, j]
            cor = "white" if valor > max_val * 0.45 else AZUL
            plt.text(j, i, str(valor), ha="center", va="center", color=cor, fontweight="bold")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_graficos, f"matriz_{nome_arquivo_limpo(nome)}.png"))
    plt.close()

def curva_aprendizado(modelo, nome, X_tr, y_tr, X_v, y_v):
    X_tr_shuf, y_tr_shuf = shuffle(X_tr, y_tr, random_state=42)
    fracoes = [0.1, 0.3, 0.5, 0.7, 1.0]

    tamanhos = []
    f1_treino_lista = []
    f1_validacao_lista = []
    total = X_tr_shuf.shape[0]

    for fracao in fracoes:
        quantidade = int(total * fracao)
        X_parcial = X_tr_shuf[:quantidade]
        y_parcial = y_tr_shuf[:quantidade]

        modelo_temp = clone(modelo)
        modelo_temp.fit(X_parcial, y_parcial)

        pred_treino = modelo_temp.predict(X_parcial)
        pred_validacao = modelo_temp.predict(X_v)

        f1_treino = f1_score(y_parcial, pred_treino, average="macro", zero_division=0)
        f1_validacao = f1_score(y_v, pred_validacao, average="macro", zero_division=0)

        tamanhos.append(quantidade)
        f1_treino_lista.append(f1_treino)
        f1_validacao_lista.append(f1_validacao)

    plt.figure(figsize=(8, 4.5))
    plt.plot(tamanhos, f1_treino_lista, marker="o", label="F1 Treino", color=AZUL)
    plt.plot(tamanhos, f1_validacao_lista, marker="o", label="F1 Validação", color=TURQUESA)
    plt.title(f"curva de aprendizado - {nome}")
    plt.xlabel("quantidade de exemplos utilizados no treinamento")
    plt.ylabel("F1 Macro")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_graficos, f"curva_{nome_arquivo_limpo(nome)}.png"))
    plt.close()

# definicao e treinamento dos modelos
modelos = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "Regressão Logistica": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "Arvore de Decisão": DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, class_weight="balanced", random_state=42),
    "SVM Linear": LinearSVC(C=0.15, class_weight="balanced", max_iter=5000)
}

resultados_validacao = {}
modelos_treinados = {}

for nome, modelo in modelos.items():
    print(f"\nprocessando: {nome}...")

    # exibir curva de aprendizado (exceto para baseline)
    if "Baseline" not in nome:
        curva_aprendizado(modelo, nome, X_train, y_train, X_val, y_val)

    # ajuste final e previsoes
    modelo.fit(X_train, y_train)
    modelos_treinados[nome] = modelo
    y_pred_val = modelo.predict(X_val)

    # metricas e graficos
    metricas = calcular_metricas(y_val, y_pred_val)
    resultados_validacao[nome] = metricas

    grafico_metricas(nome, metricas)
    grafico_matriz_confusao(nome, y_val, y_pred_val)

# comparacao e selecao do modelo final
df_resultados = pd.DataFrame.from_dict(resultados_validacao, orient="index")
candidatos = df_resultados.drop("Baseline - Dummy", errors="ignore")

# grafico comparativo F1 Macro
plt.figure(figsize=(9, 5))
candidatos["F1 macro"].sort_values().plot(kind="barh", color=TURQUESA)
plt.title("comparativo de candidatos - F1 Macro na validação")
plt.xlabel("pontuação")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(pasta_graficos, "comparativo_f1_macro.png"))
plt.close()

# fixamos a escolha no Random Forest conforme a analise de estabilidade e decisao do grupo:
modelo_selecionado_nome = "Random Forest"
modelo_selecionado = modelos_treinados[modelo_selecionado_nome]

print(f"\nmodelo selecionado metodologicamente para teste final: {modelo_selecionado_nome}")

# avaliacao no conjunto de teste
y_pred_test = modelo_selecionado.predict(X_test)
metricas_teste = calcular_metricas(y_test, y_pred_test)

print(f"\ndesempenho final: {modelo_selecionado_nome.upper()} no conjunto de teste")

# converte os nomes das classes para texto
classes_nomes_str = [str(c) for c in classes_nomes]

print(classification_report(y_test, y_pred_test, target_names=classes_nomes_str, zero_division=0))

grafico_matriz_confusao(f"teste final - {modelo_selecionado_nome}", y_test, y_pred_test) 
print(f"\nprocesso concluido com sucesso. graficos salvos na pasta '{pasta_graficos}'")