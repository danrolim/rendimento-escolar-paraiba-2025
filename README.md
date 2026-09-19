# Rendimento Escolar — Paraíba (2025)

Projeto da disciplina **Projeto Integrador III** (Ciência de Dados). Dashboard interativo e
modelos de Machine Learning sobre as **Taxas de Rendimento Escolar** (Aprovação, Reprovação
e Abandono) dos municípios da Paraíba, Ensino Fundamental e Médio, ano de 2025.

**Fonte dos dados:** [INEP — Taxas de Rendimento Escolar](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais/taxas-de-rendimento-escolar).

## Estrutura do projeto

```
data/
  raw/          # dados originais (planilha INEP + geojson dos municípios)
  processed/    # dados limpos, filtrados para a PB
src/
  data/prepare.py   # limpeza e filtragem dos dados brutos
app/
  dashboard.py      # aplicação Streamlit
notebooks/          # análise exploratória
```

## Como rodar localmente

```bash
pip install -r requirements.txt
python src/data/prepare.py   # gera data/processed/tx_rend_pb_2025.csv
streamlit run app/dashboard.py
```

## Funcionalidades do dashboard

- Filtros por localização (Urbana/Rural/Total), dependência administrativa, município,
  etapa de ensino (Fundamental/Médio) e indicador (Aprovação/Reprovação/Abandono).
- Mapa coroplético dos municípios da Paraíba.
- Rankings (top/bottom 15 municípios) e comparativos por dependência e localização.
- Tabela detalhada com exportação em CSV.
- Aba de Machine Learning:
  - **Clusterização (K-Means)** — segmenta os municípios por perfil de rendimento escolar.
  - **Classificação (Random Forest)** — prevê o nível de risco de abandono a partir da
    localização, dependência administrativa e taxas de aprovação/reprovação.

## Envio de sugestões

O final do painel tem um formulário (título curto e sugestão de até 500 palavras) cujas respostas
são gravadas em um Google Forms do autor, que avisa por e-mail a cada nova resposta. O código não
usa senha nem endereço de e-mail: só o endereço do formulário, lido de `st.secrets` (veja
`.streamlit/secrets.toml.example`). Sem essa configuração, o painel funciona normalmente e mostra
apenas um aviso no lugar do formulário.

## Deploy

Aplicação publicada no [Streamlit Community Cloud](https://streamlit.io/cloud), a partir
deste repositório GitHub (arquivo principal: `app/dashboard.py`).
