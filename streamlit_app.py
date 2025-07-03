# streamlit_app.py – Salary × GenAI Dashboard
# -----------------------------------------------------------------------------
import pandas as pd, numpy as np, re, zipfile
import plotly.express as px, streamlit as st
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from kaggle.api.kaggle_api_extended import KaggleApi

# -----------------------------------------------------------------------------
# Config / download Kaggle
DATA_DIR  = Path("data"); DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "df_survey_2024.csv"
KAGGLE_DS = "datahackers/state-of-data-brazil-20242025"
ZIP_NAME  = "state-of-data-brazil-20242025.zip"

def download_dataset():
    if DATA_PATH.exists(): return
    st.info("📦 Baixando dataset do Kaggle…")
    api = KaggleApi(); api.authenticate()
    api.dataset_download_files(KAGGLE_DS, path=str(DATA_DIR), quiet=False)
    with zipfile.ZipFile(DATA_DIR / ZIP_NAME) as zf:
        zf.extractall(str(DATA_DIR))
    next(DATA_DIR.glob("*df_survey_2024.csv")).rename(DATA_PATH)
    (DATA_DIR / ZIP_NAME).unlink(missing_ok=True)
    for p in DATA_DIR.glob("state-of-data-brazil-*"):
        if p.is_dir(): __import__("shutil").rmtree(p)

# -----------------------------------------------------------------------------
# Utils
def faixa_to_midpoint(tx: str):
    if pd.isna(tx): return np.nan
    nums = [int(x.replace(".", "")) for x in re.findall(r"R\$\s?([\d.]+)", tx)]
    if "Mais de"  in tx and nums: return nums[0]*1.2
    if "Menos de" in tx and nums: return nums[0]*0.5
    return sum(nums)/2 if len(nums)==2 else np.nan

@st.cache_data(show_spinner="🔄 Carregando dados…")
def load():
    download_dataset()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["salary_mid"] = df["2.h_faixa_salarial"].apply(faixa_to_midpoint)
    genai_cols = [c for c in df.columns if re.search(r"LLM|IA Generativa", c, re.I)]
    df["GenAI_user"] = df[genai_cols].fillna("").apply(
        lambda r: int(any(str(x).strip().lower() not in ("", "não", "nao") for x in r)), axis=1
    )
    return df.dropna(subset=["salary_mid"])

# -----------------------------------------------------------------------------
# App
st.set_page_config("Salary × GenAI", "💰", layout="wide")
df = load()

def pick(sub: str, max_u=50):
    for c in df.columns:
        if sub in c.lower() and df[c].nunique()<=max_u:
            return c
    return None

level_col, sector_col = pick("nivel"), pick("setor")
if not level_col or not sector_col:
    st.error("Colunas de nível ou setor não encontradas."); st.stop()

# --------------- sidebar filtros
st.sidebar.header("🎛️ Filtros")
levels  = sorted(df[level_col].dropna().unique())
sectors = sorted(df[sector_col].dropna().unique())

sel_levels  = st.sidebar.multiselect("Senioridade", levels,  default=levels)
sel_sectors = st.sidebar.multiselect("Setor",       sectors, default=sectors)
opt_genai   = st.sidebar.radio("GenAI", ["Todos", "Usuários", "Não usuários"])

mask = df[level_col].isin(sel_levels) & df[sector_col].isin(sel_sectors)
if opt_genai=="Usuários":       mask &= df.GenAI_user==1
elif opt_genai=="Não usuários": mask &= df.GenAI_user==0
sub = df[mask]

st.title("💰 Quem Usa GenAI Ganha Mais?")

# --------------- métricas globais
if sub.empty:
    st.warning("Sem dados para esses filtros."); st.stop()

med, mean = sub.salary_mid.median(), sub.salary_mid.mean()
mini, maxi= sub.salary_mid.min(),  sub.salary_mid.max()
c1,c2,c3,c4 = st.columns(4)
c1.metric("Mediana",f"R$ {med:,.0f}")
c2.metric("Média",  f"R$ {mean:,.0f}")
c3.metric("Mínimo", f"R$ {mini:,.0f}")
c4.metric("Máximo", f"R$ {maxi:,.0f}")

# --------------- violin
fig = px.violin(sub,y="salary_mid",x="GenAI_user",box=True,points="all",
                labels={"GenAI_user":"Usa GenAI (0=Não,1=Sim)",
                        "salary_mid":"Salário (R$)"},
                title="Distribuição salarial")
st.plotly_chart(fig, use_container_width=True)

# --------------- teste estatístico
g1,g0 = sub[sub.GenAI_user==1].salary_mid, sub[sub.GenAI_user==0].salary_mid
if len(g1)>10 and len(g0)>10:
    _, p = mannwhitneyu(g1,g0); st.caption(f"Mann-Whitney p = {p:.2e}")

# --------------- odds ratio
@st.cache_data
def odds(df_):
    m=df_[["salary_mid","GenAI_user",level_col]].dropna()
    if m.GenAI_user.nunique()<2: return None
    m["high"]=(m.salary_mid>m.salary_mid.median()).astype(int)
    X,y=m[["GenAI_user",level_col]],m.high
    mdl=Pipeline([("prep",ColumnTransformer([
            ("lvl",OneHotEncoder(handle_unknown="ignore"),[level_col]),
            ("num","passthrough",["GenAI_user"])
        ])),("clf",LogisticRegression(max_iter=1000))]).fit(X,y)
    return float(np.exp(mdl.named_steps["clf"].coef_[0][-1]))
or_ = odds(sub)
st.write(f"**OR(GenAI_user): {or_:.2f}×** prob. de estar acima da média" if or_ else
         "Sem variação GenAI neste recorte")

# -----------------------------------------------------------------------------
# 🔎 Visão por setor — média e comparação GenAI vs não
# -----------------------------------------------------------------------------
st.header("🔍 Visão por setor")

# Agrega média por setor + GenAI_user
agg = (sub.groupby([sector_col,"GenAI_user"])
          .salary_mid.mean()
          .reset_index()
          .rename(columns={"salary_mid":"mean"}))

# Mapeia 0/1 → rótulos
agg["GenAI"] = agg.GenAI_user.map({0:"Não usa GenAI",1:"Usa GenAI"})

# Ordena setor pelo total geral
order = (agg.groupby(sector_col).mean("mean")
         .sort_values("mean",ascending=False).index)
agg[sector_col] = pd.Categorical(agg[sector_col], categories=order, ordered=True)

fig2 = px.bar(
    agg, y=sector_col, x="mean", color="GenAI",
    orientation="h", barmode="group",
    labels={sector_col:"Setor", "mean":"Média (R$)"},
    height=650,
    title="Média salarial por setor — comparação GenAI × não GenAI"
)
fig2.update_traces(texttemplate="R$ %{x:,.0f}", textposition="outside")
fig2.update_layout(yaxis={"categoryorder":"array","categoryarray":order})

st.plotly_chart(fig2, use_container_width=True)

# tabela detalhada
pivot = agg.pivot(index=sector_col, columns="GenAI", values="mean").round(0)
st.dataframe(pivot.style.format("R$ {:,.0f}"), use_container_width=True)

st.caption("LBPTech © 2025 Data Hackers — dashboard demo.")
