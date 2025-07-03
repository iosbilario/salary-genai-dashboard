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
DATA_DIR   = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH  = DATA_DIR / "df_survey_2024.csv"
KAGGLE_DS  = "datahackers/state-of-data-brazil-20242025"   # slug da coleção
ZIP_NAME   = "state-of-data-brazil-20242025.zip"           # nome padrão do zip

# -------------------------------------------------------------
# Download & unzip (executa 1ª vez ou quando arquivo some)
# -------------------------------------------------------------
def download_dataset():
    if DATA_PATH.exists():
        return  # já temos o CSV

    st.info("📦 Baixando dataset (zip) do Kaggle…")
    api = KaggleApi()
    api.authenticate()               # usa KAGGLE_USERNAME / KAGGLE_KEY
    api.dataset_download_files(KAGGLE_DS, path=str(DATA_DIR), quiet=False)

    zip_path = DATA_DIR / ZIP_NAME
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(DATA_DIR))
    zip_path.unlink()  # remove o zip

# -------------------------------------------------------------
# Utilitários
# -------------------------------------------------------------
def faixa_to_midpoint(txt):
    if pd.isna(txt): return np.nan
    nums = [int(x.replace('.', '')) for x in re.findall(r'R\\$\\s?([0-9.]+)', txt)]
    if 'Mais de'  in txt and nums: return nums[0] * 1.2
    if 'Menos de' in txt and nums: return nums[0] * 0.5
    return sum(nums)/2 if len(nums) == 2 else np.nan

@st.cache_data(show_spinner="🔄 Carregando e processando dados…")
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
# Streamlit page
# -------------------------------------------------------------
st.set_page_config(page_title="Salary × GenAI", layout="wide")
df = load_data()

st.title("💰 Quem Usa GenAI Ganha Mais?")

# ----- Sidebar filtros
st.sidebar.header("🎛️ Filtros")
levels  = sorted(df["2.g_nivel"].dropna().unique())
sectors = sorted(df["2.b_setor"].dropna().unique())

sel_levels  = st.sidebar.multiselect("Senioridade", levels,  default=levels)
sel_sectors = st.sidebar.multiselect("Setor",       sectors, default=sectors)
opt_genai   = st.sidebar.radio("GenAI", ["Todos", "Usuários", "Não usuários"])

mask = df["2.g_nivel"].isin(sel_levels) & df["2.b_setor"].isin(sel_sectors)
if opt_genai == "Usuários":
    mask &= df["GenAI_user"] == 1
elif opt_genai == "Não usuários":
    mask &= df["GenAI_user"] == 0
sub = df[mask]

# ----- Métricas
col1, col2 = st.columns(2)
col1.metric("Mediana", f"R$ {sub['salary_mid'].median():,.0f}")
col2.metric("Média",   f"R$ {sub['salary_mid'].mean():,.0f}")

# ----- Violin plot
fig = px.violin(
    sub, y="salary_mid", x="GenAI_user", box=True, points="all",
    labels={"GenAI_user": "Usa GenAI (0=Não,1=Sim)", "salary_mid": "Salário (R$)"},
    title="Distribuição salarial"
)
st.plotly_chart(fig, use_container_width=True)

# ----- Teste estatístico
g1 = sub[sub.GenAI_user == 1]["salary_mid"]
g0 = sub[sub.GenAI_user == 0]["salary_mid"]
if len(g1) > 10 and len(g0) > 10:
    _, p = mannwhitneyu(g1, g0, alternative="two-sided")
    st.caption(f"Mann-Whitney p = {p:.2e}")

# ----- Odds ratio (cache p/ performance)
@st.cache_data
def odds_ratio(data):
    m = data[["salary_mid", "GenAI_user", "2.g_nivel"]].dropna()
    m["high"] = (m.salary_mid > m.salary_mid.median()).astype(int)
    X, y = m[["GenAI_user", "2.g_nivel"]], m["high"]

    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("lvl", OneHotEncoder(handle_unknown="ignore"), ["2.g_nivel"]),
            ("num", "passthrough", ["GenAI_user"])
        ])),
        ("clf", LogisticRegression(max_iter=1000))
    ]).fit(X, y)
    return float(np.exp(pipe.named_steps["clf"].coef_[0][-1]))

st.write(f"**OR(GenAI_user): {odds_ratio(sub):.2f}×** probabilidade de estar acima da mediana.")

st.caption("Dados © 2024 Data Hackers — reutilizados sob licença MIT. App demo.")
