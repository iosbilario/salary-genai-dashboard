# streamlit_app.py – Salary × GenAI Dashboard
# -------------------------------------------
import pandas as pd
import numpy as np
import re
import os
import zipfile
import plotly.express as px
import streamlit as st
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from kaggle.api.kaggle_api_extended import KaggleApi

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_DIR  = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "df_survey_2024.csv"
KAGGLE_DS = "datahackers/state-of-data-brazil-20242025"
ZIP_NAME  = "state-of-data-brazil-20242025.zip"

# ------------------------------------------------------------------
# Baixa ZIP do Kaggle caso o CSV ainda não exista
# ------------------------------------------------------------------
def download_dataset():
    if DATA_PATH.exists():
        return
    st.info("📦 Baixando dataset do Kaggle…")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DS, path=str(DATA_DIR), quiet=False)

    zip_path = DATA_DIR / ZIP_NAME
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(DATA_DIR))

    # move o CSV (nome longo) para nome curto
    csv_path = next(DATA_DIR.glob("*df_survey_2024.csv"))
    csv_path.rename(DATA_PATH)

    # limpa pasta e zip
    zip_path.unlink(missing_ok=True)
    for p in DATA_DIR.glob("state-of-data-brazil-*"):
        if p.is_dir():
            import shutil; shutil.rmtree(p)

# ------------------------------------------------------------------
# Funções utilitárias
# ------------------------------------------------------------------
def faixa_to_midpoint(txt: str):
    """Converte faixa salarial para ponto médio numérico."""
    if pd.isna(txt):
        return np.nan
    nums = [int(x.replace(".", "")) for x in re.findall(r"R\$\s?([\d.]+)", txt)]
    if "Mais de" in txt and nums:
        return nums[0] * 1.2
    if "Menos de" in txt and nums:
        return nums[0] * 0.5
    return sum(nums) / 2 if len(nums) == 2 else np.nan

@st.cache_data(show_spinner="🔄 Carregando dados…")
def load_data():
    download_dataset()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["salary_mid"] = df["2.h_faixa_salarial"].apply(faixa_to_midpoint)

    # flag GenAI
    genai_cols = [c for c in df.columns if re.search(r"LLM|IA Generativa", c, re.I)]
    df["GenAI_user"] = df[genai_cols].fillna("").apply(
        lambda r: int(any(str(x).strip().lower() not in ("", "não", "nao") for x in r)), axis=1
    )
    return df.dropna(subset=["salary_mid"])

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
st.set_page_config(page_title="Salary × GenAI", layout="wide")
df = load_data()

# ---------- Detecta colunas de nível & setor ----------
def pick_column(substring: str, max_uniques: int = 50):
    for col in df.columns:
        if substring in col.lower() and df[col].nunique() <= max_uniques:
            return col
    return None

level_col  = pick_column("nivel")
sector_col = pick_column("setor")

if not level_col or not sector_col:
    st.error(
        f"Colunas de nível ou setor não encontradas!\n"
        f"Detectado nível: {level_col} | setor: {sector_col}"
    )
    st.stop()

# ---------- Sidebar filtros ----------
st.sidebar.header("🎛️ Filtros")
levels  = sorted(df[level_col].dropna().unique())
sectors = sorted(df[sector_col].dropna().unique())

sel_levels  = st.sidebar.multiselect("Senioridade", levels, default=levels)
sel_sectors = st.sidebar.multiselect("Setor", sectors, default=sectors)
opt_genai   = st.sidebar.radio("GenAI", ["Todos", "Usuários", "Não usuários"])

mask = df[level_col].isin(sel_levels) & df[sector_col].isin(sel_sectors)
if opt_genai == "Usuários":
    mask &= df["GenAI_user"] == 1
elif opt_genai == "Não usuários":
    mask &= df["GenAI_user"] == 0

sub = df[mask]

st.title("💰 Quem Usa GenAI Ganha Mais?")

# ---------- Métricas ----------
if sub.empty:
    st.warning("Sem dados para esses filtros.")
    st.stop()

col1, col2 = st.columns(2)
col1.metric("Mediana", f"R$ {sub['salary_mid'].median():,.0f}")
col2.metric("Média",   f"R$ {sub['salary_mid'].mean():,.0f}")

# ---------- Violin plot ----------
fig = px.violin(
    sub, y="salary_mid", x="GenAI_user", box=True, points="all",
    labels={"GenAI_user": "Usa GenAI (0=Não,1=Sim)", "salary_mid": "Salário (R$)"},
    title="Distribuição salarial"
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Teste estatístico ----------
g1 = sub[sub.GenAI_user == 1]["salary_mid"]
g0 = sub[sub.GenAI_user == 0]["salary_mid"]
if len(g1) > 10 and len(g0) > 10:
    _, pval = mannwhitneyu(g1, g0, alternative="two-sided")
    st.caption(f"Mann-Whitney p = {pval:.2e}")

# ---------- Odds Ratio ----------
@st.cache_data
def odds_ratio(data: pd.DataFrame) -> float | None:
    m = data[["salary_mid", "GenAI_user", level_col]].dropna()
    if m["GenAI_user"].nunique() < 2:
        return None

    m["high"] = (m.salary_mid > m.salary_mid.median()).astype(int)
    X, y = m[["GenAI_user", level_col]], m["high"]

    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("lvl", OneHotEncoder(handle_unknown="ignore"), [level_col]),
            ("num", "passthrough", ["GenAI_user"])
        ])),
        ("clf", LogisticRegression(max_iter=1000))
    ]).fit(X, y)

    return float(np.exp(pipe.named_steps["clf"].coef_[0][-1]))

or_val = odds_ratio(sub)
if or_val is None:
    st.warning("Não há variação de GenAI suficiente neste recorte.")
else:
    st.write(f"**OR(GenAI_user): {or_val:.2f}×** chance de estar acima da mediana.")

st.caption("LbpTech © 2025 - collab with Data Hackers — dashboard demo.")
