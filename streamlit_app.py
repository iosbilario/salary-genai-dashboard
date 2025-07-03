# streamlit_app.py  – versão dinâmica (colunas auto-detectadas)
# -------------------------------------------------------------
import pandas as pd, numpy as np, re, os, zipfile
import plotly.express as px, streamlit as st
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap
from kaggle.api.kaggle_api_extended import KaggleApi

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
DATA_DIR  = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "df_survey_2024.csv"
KAGGLE_DS = "datahackers/state-of-data-brazil-20242025"
ZIP_NAME  = "state-of-data-brazil-20242025.zip"

# -------------------------------------------------------------
# Download zip + unzip (executa só se CSV não existir)
# -------------------------------------------------------------
def download_dataset():
    if DATA_PATH.exists():
        return
    st.info("📦 Baixando dataset do Kaggle…")
    api = KaggleApi(); api.authenticate()
    api.dataset_download_files(KAGGLE_DS, path=str(DATA_DIR), quiet=False)

    zip_path = DATA_DIR / ZIP_NAME
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(DATA_DIR))
    # move CSV com qualquer prefixo/espacos
    csv_path = next(DATA_DIR.glob("*df_survey_2024.csv"))
    csv_path.rename(DATA_PATH)
    zip_path.unlink()
    # remove pasta extra, se existir
    for p in DATA_DIR.glob("state-of-data-brazil-*"):
        if p.is_dir():
            import shutil; shutil.rmtree(p)

# -------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------
def faixa_to_midpoint(txt: str):
    if pd.isna(txt): return np.nan
    nums = [int(x.replace('.', '')) for x in re.findall(r'R\\$\\s?([0-9.]+)', txt)]
    if 'Mais de'  in txt and nums: return nums[0] * 1.2
    if 'Menos de' in txt and nums: return nums[0] * 0.5
    return sum(nums)/2 if len(nums) == 2 else np.nan

@st.cache_data(show_spinner="🔄 Carregando dados…")
def load_data():
    download_dataset()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["salary_mid"] = df["2.h_faixa_salarial"].apply(faixa_to_midpoint)

    genai_cols = [c for c in df.columns if re.search(r"LLM|IA Generativa", c, re.I)]
    df["GenAI_user"] = df[genai_cols].fillna("").apply(
        lambda r: int(any(str(x).strip().lower() not in ("", "não", "nao") for x in r)), axis=1
    )
    return df.dropna(subset=["salary_mid"])

# -------------------------------------------------------------
# App
# -------------------------------------------------------------
st.set_page_config(page_title="Salary × GenAI", layout="wide")
df = load_data()

# ---------- detecta colunas de nível & setor ----------
level_candidates  = [c for c in df.columns if re.search(r"(?i)nivel$",  c)]
sector_candidates = [c for c in df.columns if re.search(r"(?i)setor$", c)]
if not level_candidates or not sector_candidates:
    st.error("Colunas de nível ou setor não encontradas no dataset!")
    st.stop()
level_col  = level_candidates[0]
sector_col = sector_candidates[0]

# ---------- sidebar filtros ----------
st.sidebar.header("🎛️ Filtros")
levels  = sorted(df[level_col].dropna().unique())
sectors = sorted(df[sector_col].dropna().unique())

sel_levels  = st.sidebar.multiselect("Senioridade", levels,  default=levels)
sel_sectors = st.sidebar.multiselect("Setor",       sectors, default=sectors)
opt_genai   = st.sidebar.radio("GenAI", ["Todos", "Usuários", "Não usuários"])

mask = df[level_col].isin(sel_levels) & df[sector_col].isin(sel_sectors)
if opt_genai == "Usuários":
    mask &= df["GenAI_user"] == 1
elif opt_genai == "Não usuários":
    mask &= df["GenAI_user"] == 0
sub = df[mask]

st.title("💰 Quem Usa GenAI Ganha Mais?")

# ---------- métricas ----------
if sub.empty:
    st.warning("Sem dados para esses filtros.")
    st.stop()

col1, col2 = st.columns(2)
col1.metric("Mediana", f"R$ {sub['salary_mid'].median():,.0f}")
col2.metric("Média",   f"R$ {sub['salary_mid'].mean():,.0f}")

# ---------- violin plot ----------
fig = px.violin(
    sub, y="salary_mid", x="GenAI_user", box=True, points="all",
    labels={"GenAI_user":"Usa GenAI (0=Não,1=Sim)", "salary_mid":"Salário (R$)"},
    title="Distribuição salarial"
)
st.plotly_chart(fig, use_container_width=True)

# ---------- teste estatístico ----------
g1 = sub[sub.GenAI_user == 1]["salary_mid"]
g0 = sub[sub.GenAI_user == 0]["salary_mid"]
if len(g1) > 10 and len(g0) > 10:
    _, p = mannwhitneyu(g1, g0, alternative="two-sided")
    st.caption(f"Mann-Whitney p = {p:.2e}")

# ---------- odds ratio ----------
@st.cache_data
def odds_ratio(data: pd.DataFrame) -> float:
    m = data[["salary_mid", "GenAI_user", level_col]].dropna()
    if m["GenAI_user"].nunique() < 2:
        return np.nan
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
if np.isnan(or_val):
    st.warning("Não há variação suficiente de GenAI no recorte atual.")
else:
    st.write(f"**OR(GenAI_user): {or_val:.2f}×** chance de estar acima da mediana.")

st.caption("Dados © 2024 Data Hackers — app demo.")
