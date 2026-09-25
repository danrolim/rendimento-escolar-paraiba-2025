"""Classificador de risco elevado de abandono por perfil escolar (zona, rede e posição geográfica)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

REDES = ["Federal", "Estadual", "Municipal", "Privada"]
QUANTIL_RISCO = 0.75
FOLDS = 5
REPETICOES = 3
EMBARALHAMENTOS = 3

COLUNAS_ZONA = ["localizacao_Rural", "localizacao_Urbana"]
COLUNAS_REDE = [f"dependencia_{r}" for r in REDES]
COLUNAS_GEO = ["lat", "lon"]
GRUPOS = {
    "Zona (urbana ou rural)": COLUNAS_ZONA,
    "Rede (dependência administrativa)": COLUNAS_REDE,
    "Posição geográfica do município": COLUNAS_GEO,
}


def centroides(geojson: dict) -> dict[int, tuple[float, float]]:
    """Centro aproximado de cada município (média dos vértices do contorno externo)."""
    out = {}
    for f in geojson["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        pts = np.array([p for poly in polys for p in poly[0]])
        out[int(f["properties"]["id"])] = (pts[:, 0].mean(), pts[:, 1].mean())
    return out


def montar_base(df: pd.DataFrame, geojson: dict, coluna_abandono: str) -> tuple[pd.DataFrame, float]:
    """Perfis (município, zona, rede) com abandono conhecido, atributos e alvo 'risco elevado'."""
    d = df[df["localizacao"].isin(["Urbana", "Rural"]) & df["dependencia"].isin(REDES)]
    d = d.dropna(subset=[coluna_abandono]).reset_index(drop=True)
    corte = float(d[coluna_abandono].quantile(QUANTIL_RISCO))

    X = pd.get_dummies(d[["localizacao", "dependencia"]]).astype(float)
    X = X.reindex(columns=COLUNAS_ZONA + COLUNAS_REDE, fill_value=0.0)
    cent = centroides(geojson)
    X["lon"] = d["codigo_municipio"].map(lambda c: cent[c][0])
    X["lat"] = d["codigo_municipio"].map(lambda c: cent[c][1])
    X["codigo_municipio"] = d["codigo_municipio"]
    X["risco_elevado"] = (d[coluna_abandono] > corte).astype(int)
    return X, corte


def novo_modelo() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=42)


def avaliar(base: pd.DataFrame) -> dict:
    """Validação cruzada agrupada por município (perfis do mesmo município nunca ficam em treino e teste)."""
    y = base["risco_elevado"].to_numpy()
    grupos = base["codigo_municipio"].to_numpy()
    X = base.drop(columns=["codigo_municipio", "risco_elevado"])
    rng = np.random.default_rng(0)

    metricas, queda = [], {nome: [] for nome in GRUPOS}
    predicoes = np.zeros(len(y), dtype=int)
    for rep in range(REPETICOES):
        for tr, te in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=rep).split(X, y, grupos):
            modelo = novo_modelo().fit(X.iloc[tr], y[tr])
            prob = modelo.predict_proba(X.iloc[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            if rep == 0:
                predicoes[te] = pred
            auc = roc_auc_score(y[te], prob)
            metricas.append({
                "auc": auc,
                "recall": recall_score(y[te], pred),
                "precisao": precision_score(y[te], pred, zero_division=0),
            })
            for nome, cols in GRUPOS.items():  # embaralha o grupo inteiro e mede quanto o AUC cai
                for _ in range(EMBARALHAMENTOS):
                    Xp = X.iloc[te].copy()
                    Xp[cols] = Xp[cols].to_numpy()[rng.permutation(len(te))]
                    queda[nome].append(auc - roc_auc_score(y[te], modelo.predict_proba(Xp)[:, 1]))

    m = pd.DataFrame(metricas)
    peso = pd.Series({n: max(np.mean(v), 0.0) for n, v in queda.items()})
    return {
        "n_perfis": len(y),
        "n_municipios": int(pd.Series(grupos).nunique()),
        "prevalencia": float(y.mean()),
        "media": m.mean().to_dict(),
        "desvio": m.std().to_dict(),
        "matriz": confusion_matrix(y, predicoes, labels=[0, 1]),
        "importancias_pct": (peso / peso.sum() * 100).sort_values() if peso.sum() > 0 else peso,
    }
