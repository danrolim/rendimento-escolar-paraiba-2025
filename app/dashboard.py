"""Dashboard de Taxas de Rendimento Escolar - Municipios da Paraiba (2025).

Fonte dos dados: INEP - Taxas de Rendimento Escolar por Municipio.
"""

import json
import time
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sugestoes import MAX_CARACTERES, MAX_PALAVRAS, MAX_TITULO, contar_palavras, enviar, montar_dados
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

SUFFIX_PRETTY = {
    "fund_total": "Fundamental Total",
    "fund_anos_iniciais": "Fundamental Anos Iniciais",
    "fund_anos_finais": "Fundamental Anos Finais",
    "fund_1_ano": "Fundamental 1º Ano",
    "fund_2_ano": "Fundamental 2º Ano",
    "fund_3_ano": "Fundamental 3º Ano",
    "fund_4_ano": "Fundamental 4º Ano",
    "fund_5_ano": "Fundamental 5º Ano",
    "fund_6_ano": "Fundamental 6º Ano",
    "fund_7_ano": "Fundamental 7º Ano",
    "fund_8_ano": "Fundamental 8º Ano",
    "fund_9_ano": "Fundamental 9º Ano",
    "medio_total": "Médio Total",
    "medio_1_serie": "Médio 1ª Série",
    "medio_2_serie": "Médio 2ª Série",
    "medio_3_serie": "Médio 3ª Série",
    "medio_4_serie": "Médio 4ª Série",
    "medio_nao_seriado": "Médio Não Seriado",
}

CATEGORY_PRETTY = {"localizacao": "Localização", "dependencia": "Dependência"}


def pretty_label(col: str) -> str:
    """Converte um nome de coluna de taxa (ex.: aprovacao_fund_total) em rótulo legível."""
    prefix, _, suffix = col.partition("_")
    if prefix in RATE_LABELS and suffix in SUFFIX_PRETTY:
        return f"{RATE_LABELS[prefix]} {SUFFIX_PRETTY[suffix]}"
    return col


def pretty_feature_name(name: str) -> str:
    """Converte nomes de features one-hot (ex.: localizacao_Rural) em rótulos legíveis."""
    for code, label in CATEGORY_PRETTY.items():
        if name.startswith(f"{code}_"):
            return f"{label}: {name[len(code) + 1:]}"
    return pretty_label(name)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_geojson() -> dict:
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def rate_column(taxa: str, etapa: str, detalhe: str) -> str:
    return f"{taxa}_{ETAPA_DETALHE[etapa][detalhe]}"


FEATURE_COLS = [
    "aprovacao_fund_total", "reprovacao_fund_total", "abandono_fund_total",
    "aprovacao_medio_total", "reprovacao_medio_total", "abandono_medio_total",
]


@st.cache_data
def run_clustering(k: int) -> pd.DataFrame:
    dados = load_data()
    base = dados[(dados["localizacao"] == "Total") & (dados["dependencia"] == "Total")]
    base = base.dropna(subset=FEATURE_COLS).copy()

    X = StandardScaler().fit_transform(base[FEATURE_COLS])
    base["cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).labels_.astype(str)

    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    base["pca_1"], base["pca_2"] = coords[:, 0], coords[:, 1]
    return base


@st.cache_data
def train_classifier(target_col: str) -> dict | None:
    dados = load_data()
    clf_df = dados[
        dados["localizacao"].isin(["Urbana", "Rural"])
        & dados["dependencia"].isin(["Federal", "Estadual", "Municipal", "Privada"])
    ].dropna(subset=[target_col]).copy()

    risco_codes, risco_bins = pd.qcut(clf_df[target_col], q=3, retbins=True, duplicates="drop", labels=False)
    labels_disponiveis = ["Baixo", "Médio", "Alto"][-(len(risco_bins) - 1):]
    clf_df["risco"] = pd.Categorical.from_codes(risco_codes, categories=labels_disponiveis)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_clf = encoder.fit_transform(clf_df[["localizacao", "dependencia"]])
    y_clf = clf_df["risco"]

    if y_clf.nunique() < 2 or len(clf_df) < 20:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    labels = sorted(y_clf.unique())
    feature_names = [pretty_feature_name(n) for n in encoder.get_feature_names_out(["localizacao", "dependencia"])]
    return {
        "acc": accuracy_score(y_test, y_pred),
        "labels": labels,
        "cm": confusion_matrix(y_test, y_pred, labels=labels),
        "importancias_pct": pd.Series(rf.feature_importances_ * 100, index=feature_names).sort_values(),
    }


def chave_alfabetica(texto: str) -> str:
    """Chave de ordenação que ignora acentos (ordem alfabética correta em português)."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii").lower()


def destaque_municipio(dados: pd.DataFrame, coluna: str, modo: str) -> str:
    """Formata o município de destaque (maior/menor taxa), indicando empates."""
    validos = dados.dropna(subset=[coluna])
    if validos.empty:
        return "-"
    valor = validos[coluna].max() if modo == "max" else validos[coluna].min()
    empatados = sorted(validos.loc[validos[coluna] == valor, "municipio"], key=chave_alfabetica)
    texto = empatados[0]
    restantes = len(empatados) - 1
    if restantes == 1:
        texto += " e outro município"
    elif restantes > 1:
        texto += f" e outros {restantes} municípios"
    return f"{texto} ({valor:.1f}%)"


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

k1, k4 = st.columns(2)
k1.metric(f"Média estadual — {RATE_LABELS[taxa]}", f"{map_df[col].mean():.1f}%")
k4.metric("Municípios no filtro", f"{map_df['codigo_municipio'].nunique()}")

k2, k3 = st.columns(2)
with k2:
    st.markdown("**Município com maior taxa**")
    st.markdown(destaque_municipio(map_df, col, "max"))
with k3:
    st.markdown("**Município com menor taxa**")
    st.markdown(destaque_municipio(map_df, col, "min"))

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
        st.subheader(f"15 municípios com maior - {RATE_LABELS[taxa]}")
        top = map_df.nlargest(15, col)
        st.plotly_chart(
            px.bar(top, x=col, y="municipio", orientation="h", labels={col: f"{RATE_LABELS[taxa]} (%)"})
            .update_yaxes(categoryorder="total ascending"),
            use_container_width=True,
        )

    with c2:
        st.subheader(f"15 municípios com menor - {RATE_LABELS[taxa]}")
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
    tabela_df = (
        filtered[["municipio", "localizacao", "dependencia", col]]
        .sort_values(col, ascending=False)
        .rename(columns={
            "municipio": "Município",
            "localizacao": "Localização",
            "dependencia": "Dependência Administrativa",
            col: pretty_label(col),
        })
    )
    st.dataframe(tabela_df, use_container_width=True, height=600)
    st.download_button(
        "Baixar CSV filtrado",
        tabela_df.to_csv(index=False).encode("utf-8"),
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

    k = st.slider("Número de clusters (k)", min_value=2, max_value=6, value=3)
    cluster_base = run_clustering(k)
    feature_cols = FEATURE_COLS

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

    st.subheader("O que cada cluster significa")

    media_estadual = cluster_base[feature_cols].mean()
    perfil = cluster_base.groupby("cluster")[feature_cols].mean()

    # Quanto cada cluster desvia da média estadual (positivo = melhor que a média,
    # já ajustado para que aprovação alta e reprovação/abandono baixos sejam "positivos").
    sinal = {c: (1 if c.startswith("aprovacao") else -1) for c in feature_cols}
    desvio = perfil.apply(lambda row: sum((row[c] - media_estadual[c]) * sinal[c] for c in feature_cols), axis=1)
    ranking = desvio.rank(ascending=False, method="first")

    def classificar(posicao: float, total: int) -> tuple[str, str]:
        fracao = (total - posicao) / (total - 1) if total > 1 else 1
        if fracao >= 2 / 3:
            return "🟢", "Bom desempenho geral"
        if fracao <= 1 / 3:
            return "🔴", "Necessita mais atenção"
        return "🟡", "Desempenho intermediário"

    def maior_desvio(cluster_id: str) -> str:
        deltas = {c: perfil.loc[cluster_id, c] - media_estadual[c] for c in feature_cols}
        destaque_col = max(deltas, key=lambda c: abs(deltas[c]))
        delta = deltas[destaque_col]
        if abs(delta) < 0.1:
            return "praticamente igual à média estadual em todos os indicadores"
        direcao = "acima" if delta > 0 else "abaixo"
        return f"{pretty_label(destaque_col)} {abs(delta):.1f} p.p. {direcao} da média estadual"

    situacao, explicacao = {}, {}
    for cluster_id in perfil.index:
        icone, rotulo = classificar(ranking[cluster_id], len(perfil))
        situacao[cluster_id] = f"{icone} {rotulo}"
        explicacao[cluster_id] = maior_desvio(cluster_id)

    st.caption(
        "Cada cluster é comparado com a média estadual dos indicadores usados na "
        "clusterização. 🟢 = desempenho acima da média estadual; 🟡 = próximo da média; "
        "🔴 = pontos de atenção abaixo da média estadual."
    )
    for cluster_id in perfil.index:
        st.markdown(f"- **Cluster {cluster_id} — {situacao[cluster_id]}**: {explicacao[cluster_id]}.")

    st.subheader("Perfil médio de cada cluster")
    tabela_perfil = perfil.copy()
    tabela_perfil.insert(0, "Situação", pd.Series(situacao))
    tabela_perfil.loc["Média estadual"] = pd.concat([pd.Series({"Situação": "⚪ Referência estadual"}), media_estadual])
    tabela_perfil = tabela_perfil.rename(columns={c: pretty_label(c) for c in feature_cols})
    st.dataframe(tabela_perfil.round(1), use_container_width=True)

    st.divider()

    st.header("Classificação de risco de abandono")
    st.markdown(
        "Este modelo aprende, a partir dos dados reais dos 223 municípios da Paraíba, se o "
        "perfil escolar, ou seja, a localização (urbana ou rural) e a rede administrativa "
        "(federal, estadual, municipal ou privada), já indica um risco maior ou menor de "
        "abandono escolar."
    )

    clf_etapa = st.radio("Etapa para classificação", list(ETAPA_DETALHE.keys()), key="clf_etapa")
    target_col = rate_column("abandono", clf_etapa, "Total")

    resultado = train_classifier(target_col)

    if resultado is None:
        st.warning("Dados insuficientes para treinar o classificador com esses filtros.")
    else:
        acc, labels, cm, importancias_pct = (
            resultado["acc"], resultado["labels"], resultado["cm"], resultado["importancias_pct"]
        )

        mc1, mc2 = st.columns([1, 1])
        with mc1:
            st.metric("Acurácia (conjunto de teste)", f"{acc:.0%}")
            st.plotly_chart(
                px.imshow(
                    cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Previsto", y="Real", color="Contagem"),
                    title="Matriz de confusão",
                ),
                use_container_width=True,
            )
            for i, lbl in enumerate(labels):
                total = cm[i].sum()
                if total == 0:
                    continue
                acertos = cm[i, i]
                st.markdown(
                    f"- Dos **{total}** perfis escolares que realmente tiveram risco **{lbl}**, "
                    f"o modelo acertou **{acertos}** ({acertos / total:.0%})."
                )
        with mc2:
            st.plotly_chart(
                px.bar(
                    importancias_pct, orientation="h", title="Peso de cada variável na decisão do modelo",
                    labels={"value": "Peso na decisão do modelo (%)", "index": ""},
                ),
                use_container_width=True,
            )

# -------------------------------------------------------------- Sugestões --
INTERVALO_ENVIO_S = 60

st.divider()
st.header("Envie sua sugestão")

# O endereço do formulário fica em st.secrets (o repositório é público).
try:
    cfg_form = dict(st.secrets["sugestoes"])
    envio_configurado = all(k in cfg_form for k in ("url", "campo_titulo", "campo_sugestao"))
except (KeyError, FileNotFoundError):
    cfg_form, envio_configurado = {}, False

if not envio_configurado:
    st.info("O envio de sugestões ainda não está disponível neste painel.")
else:
    st.caption(
        "Conte o que pode melhorar neste painel. A sugestão é recebida pelo autor do projeto e não é "
        "publicada. Não informe dados pessoais."
    )
    with st.form("form_sugestao"):
        titulo = st.text_input("Título (curto)", max_chars=MAX_TITULO)
        sugestao = st.text_area(f"Sua sugestão (até {MAX_PALAVRAS} palavras)", height=200, max_chars=MAX_CARACTERES)
        enviado = st.form_submit_button("Enviar sugestão")

    if enviado:
        espera = INTERVALO_ENVIO_S - (time.time() - st.session_state.get("ultimo_envio", 0))
        if not titulo.strip() or not sugestao.strip():
            st.warning("Preencha o título e a sugestão.")
        elif contar_palavras(sugestao) > MAX_PALAVRAS:
            st.warning(f"A sugestão tem {contar_palavras(sugestao)} palavras. O limite é {MAX_PALAVRAS}.")
        elif espera > 0:
            st.warning(f"Aguarde {int(espera) + 1} segundos para enviar outra sugestão.")
        else:
            try:
                dados = montar_dados(titulo, sugestao, cfg_form["campo_titulo"], cfg_form["campo_sugestao"])
                enviar(cfg_form["url"], dados)
            except OSError:
                st.error("Não foi possível enviar a sugestão agora. Tente novamente mais tarde.")
            else:
                st.session_state["ultimo_envio"] = time.time()
                st.success("Obrigado! Sua sugestão foi enviada.")
