"""Compara alternativas para o classificador de risco elevado de abandono.

Uso:
    python src/ml/avaliar_classificador.py            # comparação de modelos e variáveis
    python src/ml/avaliar_classificador.py --ajuste   # inclui ajuste de hiperparâmetros (validação aninhada)

Validação cruzada de 5 partes repetida 3 vezes, com os perfis de um mesmo município sempre na mesma parte.
O resultado também é salvo em data/processed/avaliacao_classificador.csv.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
from classificador import COLUNAS_GEO, COLUNAS_REDE, COLUNAS_ZONA, montar_base, novo_modelo  # noqa: E402

warnings.filterwarnings("ignore")
PERFIL = COLUNAS_ZONA + COLUNAS_REDE
MODELOS = {
    "Linha de base (sempre 'não elevado')": (PERFIL, lambda: DummyClassifier(strategy="most_frequent")),
    "Random Forest, só perfil, sem balanceamento": (PERFIL, lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42)),
    "Random Forest, só perfil, balanceado": (PERFIL, novo_modelo),
    "Regressão logística, perfil e posição, balanceada": (
        PERFIL + COLUNAS_GEO,
        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    ),
    "Random Forest, perfil e posição, balanceado (usado no painel)": (PERFIL + COLUNAS_GEO, novo_modelo),
}


def validar(base, colunas, fabrica):
    y, g, X = base["risco_elevado"].to_numpy(), base["codigo_municipio"].to_numpy(), base[colunas]
    linhas = []
    for rep in range(3):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g):
            m = fabrica().fit(X.iloc[tr], y[tr])
            pred, prob = m.predict(X.iloc[te]), m.predict_proba(X.iloc[te])[:, 1]
            linhas.append({
                "AUC": roc_auc_score(y[te], prob), "Acurácia balanceada": balanced_accuracy_score(y[te], pred),
                "Recall": recall_score(y[te], pred), "Precisão": precision_score(y[te], pred, zero_division=0),
                "Acurácia": (pred == y[te]).mean(),
            })
    return pd.DataFrame(linhas).mean().round(3).to_dict()


def ajuste(base):
    """Ajusta max_depth e min_samples_leaf com validação aninhada e compara com os parâmetros fixos."""
    y, g, X = base["risco_elevado"].to_numpy(), base["codigo_municipio"].to_numpy(), base[PERFIL + COLUNAS_GEO]
    grade = {"max_depth": [3, 6, None], "min_samples_leaf": [1, 5, 10]}
    ajustado, fixo = [], []
    for rep in range(3):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g):
            gs = GridSearchCV(
                RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
                grade, scoring="roc_auc", cv=StratifiedGroupKFold(3, shuffle=True, random_state=0),
            ).fit(X.iloc[tr], y[tr], groups=g[tr])
            ajustado.append(roc_auc_score(y[te], gs.predict_proba(X.iloc[te])[:, 1]))
            fixo.append(roc_auc_score(y[te], novo_modelo().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]))
    return round(float(np.mean(ajustado)), 3), round(float(np.mean(fixo)), 3)


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "tx_rend_pb_2025.csv")
    geo = json.load(open(ROOT / "data" / "raw" / "geojs-25-mun.json", encoding="utf-8"))
    tabela = []
    for etapa, coluna in (("Fundamental", "abandono_fund_total"), ("Médio", "abandono_medio_total")):
        base, corte = montar_base(df, geo, coluna)
        print(f"\n{etapa}: {len(base)} perfis, risco elevado = abandono acima de {corte:.1f}% ({base['risco_elevado'].mean():.0%} dos perfis)")
        for nome, (colunas, fabrica) in MODELOS.items():
            r = validar(base, colunas, fabrica)
            tabela.append({"Etapa": etapa, "Modelo": nome, **r})
        if "--ajuste" in sys.argv:
            aj, fx = ajuste(base)
            print(f"  Ajuste de hiperparâmetros (AUC aninhado): ajustado {aj} contra parâmetros fixos {fx}")
    out = pd.DataFrame(tabela)
    print("\n" + out.to_string(index=False))
    out.to_csv(ROOT / "data" / "processed" / "avaliacao_classificador.csv", index=False)


if __name__ == "__main__":
    main()
