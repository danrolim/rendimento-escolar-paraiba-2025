"""Dashboard de Taxas de Rendimento Escolar - Municipios da Paraiba (2025).

Fonte dos dados: INEP - Taxas de Rendimento Escolar por Municipio.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "tx_rend_pb_2025.csv"
GEOJSON_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "geojs-25-mun.json"

RATE_LABELS = {"aprovacao": "Aprovação", "reprovacao": "Reprovação", "abandono": "Abandono"}

ETAPA_DETALHE = {
    "Fundamental": {
        "Total": "fund_total",
        "Anos Iniciais": "fund_anos_iniciais",
        "Anos Finais": "fund_anos_finais",
    },
    "Médio": {
        "Total": "medio_total",
        "1ª série": "medio_1_serie",
        "2ª série": "medio_2_serie",
        "3ª série": "medio_3_serie",
    },
}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_geojson() -> dict:
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def rate_column(taxa: str, etapa: str, detalhe: str) -> str:
    return f"{taxa}_{ETAPA_DETALHE[etapa][detalhe]}"


st.set_page_config(page_title="Rendimento Escolar - PB (2025)", layout="wide")
st.title("Taxas de Rendimento Escolar — Municípios da Paraíba (2025)")
st.caption("Fonte: INEP — Taxas de Rendimento Escolar por Município, 2025.")

df = load_data()

# ---------------------------------------------------------------- Sidebar --
st.sidebar.header("Filtros")

localizacoes = sorted(df["localizacao"].dropna().unique())
dependencias = sorted(df["dependencia"].dropna().unique())
municipios = sorted(df["municipio"].dropna().unique())

sel_localizacao = st.sidebar.multiselect("Localização", localizacoes, default=["Total"])
sel_dependencia = st.sidebar.multiselect("Dependência Administrativa", dependencias, default=["Total"])
sel_municipios = st.sidebar.multiselect("Município (opcional)", municipios)

etapa = st.sidebar.radio("Etapa de ensino", list(ETAPA_DETALHE.keys()))
detalhe = st.sidebar.selectbox("Detalhamento", list(ETAPA_DETALHE[etapa].keys()))
taxa = st.sidebar.radio("Indicador", list(RATE_LABELS.keys()), format_func=lambda k: RATE_LABELS[k])

col = rate_column(taxa, etapa, detalhe)

filtered = df[df["localizacao"].isin(sel_localizacao) & df["dependencia"].isin(sel_dependencia)]
if sel_municipios:
    filtered = filtered[filtered["municipio"].isin(sel_municipios)]

if filtered.empty:
    st.warning("Nenhum registro para os filtros selecionados.")
    st.stop()

# ------------------------------------------------------------------ KPIs --
map_df = filtered.groupby(["codigo_municipio", "municipio"], as_index=False)[col].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Média estadual — {RATE_LABELS[taxa]}", f"{map_df[col].mean():.1f}%")
k2.metric("Município com maior taxa", map_df.loc[map_df[col].idxmax(), "municipio"] if map_df[col].notna().any() else "-")
k3.metric("Município com menor taxa", map_df.loc[map_df[col].idxmin(), "municipio"] if map_df[col].notna().any() else "-")
k4.metric("Municípios no filtro", f"{map_df['codigo_municipio'].nunique()}")

st.divider()

tab_mapa, tab_comparativos, tab_tabela, tab_ml = st.tabs(
    ["🗺️ Mapa", "📊 Comparativos", "📋 Tabela", "🤖 Machine Learning"]
)

# ------------------------------------------------------------------ Mapa --
with tab_mapa:
    geojson = load_geojson()
    fig = px.choropleth(
        map_df,
        geojson=geojson,
        locations="codigo_municipio",
        featureidkey="properties.id",
        color=col,
        color_continuous_scale="RdYlGn" if taxa == "aprovacao" else "YlOrRd",
        hover_name="municipio",
        labels={col: f"{RATE_LABELS[taxa]} (%)"},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=550)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------ Comparativos --
with tab_comparativos:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"Top 15 municípios — {RATE_LABELS[taxa]}")
        top = map_df.nlargest(15, col)
        st.plotly_chart(
            px.bar(top, x=col, y="municipio", orientation="h", labels={col: f"{RATE_LABELS[taxa]} (%)"})
            .update_yaxes(categoryorder="total ascending"),
            use_container_width=True,
        )

    with c2:
        st.subheader(f"Bottom 15 municípios — {RATE_LABELS[taxa]}")
        bottom = map_df.nsmallest(15, col)
        st.plotly_chart(
            px.bar(bottom, x=col, y="municipio", orientation="h", labels={col: f"{RATE_LABELS[taxa]} (%)"})
            .update_yaxes(categoryorder="total descending"),
            use_container_width=True,
        )

    st.subheader(f"Distribuição de {RATE_LABELS[taxa]} por Dependência Administrativa")
    box_df = df[df["localizacao"].isin(sel_localizacao) & df["dependencia"].isin(dependencias)]
    box_df = box_df[box_df["dependencia"] != "Total"]
    st.plotly_chart(
        px.box(box_df, x="dependencia", y=col, labels={col: f"{RATE_LABELS[taxa]} (%)", "dependencia": "Dependência"}),
        use_container_width=True,
    )

    st.subheader(f"Distribuição de {RATE_LABELS[taxa]} por Localização")
    box_loc_df = df[df["dependencia"].isin(sel_dependencia)]
    box_loc_df = box_loc_df[box_loc_df["localizacao"] != "Total"]
    st.plotly_chart(
        px.box(box_loc_df, x="localizacao", y=col, labels={col: f"{RATE_LABELS[taxa]} (%)", "localizacao": "Localização"}),
        use_container_width=True,
    )

# ----------------------------------------------------------------- Tabela --
with tab_tabela:
    st.dataframe(
        filtered[["municipio", "localizacao", "dependencia", col]].sort_values(col, ascending=False),
        use_container_width=True,
        height=600,
    )
    st.download_button(
        "Baixar CSV filtrado",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="tx_rend_pb_filtrado.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------- ML --
with tab_ml:
    st.header("Segmentação de municípios (Clusterização)")
    st.caption(
        "K-Means agrupa os municípios por perfil de aprovação, reprovação e abandono "
        "(Fundamental e Médio, totais por município)."
    )

    cluster_base = df[(df["localizacao"] == "Total") & (df["dependencia"] == "Total")].copy()
    feature_cols = [
        "aprovacao_fund_total", "reprovacao_fund_total", "abandono_fund_total",
        "aprovacao_medio_total", "reprovacao_medio_total", "abandono_medio_total",
    ]
    cluster_base = cluster_base.dropna(subset=feature_cols)

    k = st.slider("Número de clusters (k)", min_value=2, max_value=6, value=3)

    X = StandardScaler().fit_transform(cluster_base[feature_cols])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    cluster_base["cluster"] = kmeans.labels_.astype(str)

    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    cluster_base["pca_1"], cluster_base["pca_2"] = coords[:, 0], coords[:, 1]

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.plotly_chart(
            px.scatter(
                cluster_base, x="pca_1", y="pca_2", color="cluster", hover_name="municipio",
                labels={"pca_1": "Componente 1", "pca_2": "Componente 2"},
                title="Municípios projetados em 2D (PCA)",
            ),
            use_container_width=True,
        )
    with cc2:
        geojson = load_geojson()
        fig_cluster = px.choropleth(
            cluster_base, geojson=geojson, locations="codigo_municipio", featureidkey="properties.id",
            color="cluster", hover_name="municipio", title="Clusters no mapa",
        )
        fig_cluster.update_geos(fitbounds="locations", visible=False)
        fig_cluster.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("Perfil médio de cada cluster")
    st.dataframe(cluster_base.groupby("cluster")[feature_cols].mean().round(1), use_container_width=True)

    st.divider()

    st.header("Classificação de risco de abandono")
    st.caption(
        "Random Forest prevê o nível de risco de abandono (Baixo/Médio/Alto) de uma "
        "combinação Localização × Dependência a partir dessas variáveis e das taxas de "
        "aprovação/reprovação."
    )

    clf_etapa = st.radio("Etapa para classificação", list(ETAPA_DETALHE.keys()), key="clf_etapa")
    target_col = rate_column("abandono", clf_etapa, "Total")
    aprov_col = rate_column("aprovacao", clf_etapa, "Total")
    reprov_col = rate_column("reprovacao", clf_etapa, "Total")

    clf_df = df[
        df["localizacao"].isin(["Urbana", "Rural"]) & df["dependencia"].isin(["Federal", "Estadual", "Municipal", "Privada"])
    ].dropna(subset=[target_col, aprov_col, reprov_col]).copy()

    risco_codes, risco_bins = pd.qcut(clf_df[target_col], q=3, retbins=True, duplicates="drop", labels=False)
    all_labels = ["Baixo", "Médio", "Alto"]
    labels_disponiveis = all_labels[-(len(risco_bins) - 1):]
    clf_df["risco"] = pd.Categorical.from_codes(risco_codes, categories=labels_disponiveis)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_features = encoder.fit_transform(clf_df[["localizacao", "dependencia"]])
    num_features = clf_df[[aprov_col, reprov_col]].to_numpy()
    X_clf = np.hstack([cat_features, num_features])
    y_clf = clf_df["risco"]

    if y_clf.nunique() < 2 or len(clf_df) < 20:
        st.warning("Dados insuficientes para treinar o classificador com esses filtros.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf
        )
        rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mc1, mc2 = st.columns([1, 1])
        with mc1:
            st.metric("Acurácia (conjunto de teste)", f"{acc:.0%}")
            labels = sorted(y_clf.unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            st.plotly_chart(
                px.imshow(
                    cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Previsto", y="Real", color="Contagem"),
                    title="Matriz de confusão",
                ),
                use_container_width=True,
            )
        with mc2:
            feature_names = list(encoder.get_feature_names_out(["localizacao", "dependencia"])) + [aprov_col, reprov_col]
            importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
            st.plotly_chart(
                px.bar(importances, orientation="h", title="Importância das variáveis", labels={"value": "Importância", "index": ""}),
                use_container_width=True,
            )
