"""Le a planilha bruta do INEP (Taxas de Rendimento Escolar por Municipio - 2025),
filtra os municipios da Paraiba (PB) e gera um CSV limpo para uso no dashboard e no ML.
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/tx_rend_municipios_2025.xlsx")
OUT_PATH = Path("data/processed/tx_rend_pb_2025.csv")

SHEET_NAME = "MUNICIPIOS "
HEADER_ROW = 8  # linha (0-indexed) com os codigos de coluna (NU_ANO_CENSO, ...)

# Mapeia os codigos de coluna do INEP para nomes legiveis.
# Blocos: 1_CAT_* = Aprovacao, 2_CAT_* = Reprovacao, 3_CAT_* = Abandono
RATE_PREFIXES = {"1_CAT": "aprovacao", "2_CAT": "reprovacao", "3_CAT": "abandono"}

SUFFIX_LABELS = {
    "FUN": "fund_total",
    "FUN_AI": "fund_anos_iniciais",
    "FUN_AF": "fund_anos_finais",
    "FUN_01": "fund_1_ano",
    "FUN_02": "fund_2_ano",
    "FUN_03": "fund_3_ano",
    "FUN_04": "fund_4_ano",
    "FUN_05": "fund_5_ano",
    "FUN_06": "fund_6_ano",
    "FUN_07": "fund_7_ano",
    "FUN_08": "fund_8_ano",
    "FUN_09": "fund_9_ano",
    "MED": "medio_total",
    "MED_01": "medio_1_serie",
    "MED_02": "medio_2_serie",
    "MED_03": "medio_3_serie",
    "MED_04": "medio_4_serie",
    "MED_NS": "medio_nao_seriado",
}

BASE_COLUMNS = {
    "NU_ANO_CENSO": "ano",
    "NO_REGIAO": "regiao",
    "SG_UF": "uf",
    "CO_MUNICIPIO": "codigo_municipio",
    "NO_MUNICIPIO": "municipio",
    "NO_CATEGORIA": "localizacao",
    "NO_DEPENDENCIA": "dependencia",
}


def build_rename_map(columns: list[str]) -> dict[str, str]:
    rename = dict(BASE_COLUMNS)
    for col in columns:
        if col in rename or col is None:
            continue
        for code, rate_name in RATE_PREFIXES.items():
            if col.startswith(code):
                suffix = col[len(code) :].lstrip("_")
                label = SUFFIX_LABELS.get(suffix)
                if label:
                    rename[col] = f"{rate_name}_{label}"
    return rename


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(RAW_PATH, sheet_name=SHEET_NAME, header=HEADER_ROW)
    df = df.loc[:, df.columns.notna()]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = build_rename_map(list(df.columns))
    df = df.rename(columns=rename_map)

    df = df[df["uf"] == "PB"].copy()
    df = df.dropna(subset=["codigo_municipio"])

    rate_cols = [c for c in df.columns if c.split("_")[0] in RATE_PREFIXES.values()]
    for col in rate_cols:
        df[col] = pd.to_numeric(df[col].replace("--", pd.NA), errors="coerce")

    df["codigo_municipio"] = df["codigo_municipio"].astype(int)
    df["ano"] = df["ano"].astype(int)

    ordered_cols = list(BASE_COLUMNS.values()) + sorted(rate_cols)
    return df[ordered_cols].reset_index(drop=True)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = clean(load_raw())
    df.to_csv(OUT_PATH, index=False)
    print(f"Gerado {OUT_PATH} com {len(df)} linhas e {len(df.columns)} colunas.")


if __name__ == "__main__":
    main()
