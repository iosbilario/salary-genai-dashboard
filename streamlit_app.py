# streamlit_app.py – Salary × GenAI Dashboard
# -----------------------------------------------------------------------------
import pandas as pd, numpy as np, re, zipfile, streamlit as st, plotly.express as px
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from kaggle.api.kaggle_api_extended import KaggleApi

# -----------------------------------------------------------------------------
# Config & download
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
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
def faixa_to_mid(tx):
    if pd.isna(tx): return np.nan
    nums = [int(x.replace(".", "")) for x in re.findall(r"R\$\s?([\d.]+)", tx)]
    if "Mais de"  in tx and nums: return nums[0]*1.2
    if "Menos de" in tx and nums: return nums[0]*0.5
    return sum(nums)/2 if len(nums)==2 else np.nan

@st.cache_data(show_spinner="🔄 carregando…")
def load():
    download_dataset()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["salary_mid"] = df["2.h_faixa_salarial"].apply(faixa_to_mid)

    genai_cols = [c for c in df.columns if re.search(r"LLM|IA Generativa", c, re.I)]
    df["GenAI_user"] = df[genai_cols].fillna("").apply(
        lambda r: int(any(str(x).strip().lower() not in ("", "não", "nao") for x in r)), axis=1
    )
    return df.dropna(subset=["salary_mid"])

# -----------------------------------------------------------------------------
# App
st.set_page_config("Salary × GenAI", "💰", layout="wide")
df = load()

def pick(sub, max_u=50):
    for c in df.columns:
        if sub in c.lower() and df[c].nunique()<=max_u:
            return c
    return None

level_col, sector_col = pick("nivel"), pick("setor")
if not level_col or not sector_col:
    st.error("Colunas nível/​setor não encontradas!"); st.stop()

# Filtros
st.sidebar.header("🎛️ Filtros")
levels  = sorted(df[level_col].dropna().unique())
sectors = sorted(df[sector_col].dropna().unique())

sel_levels  = st.sidebar.multiselect("Senioridade", levels,  default=levels)
sel_sectors = st.sidebar.multiselect("Setor",       sectors, default=sectors)
opt_genai   = st.sidebar.radio("GenAI", ["Todos", "Usuários", "Não usuários"])

mask = df[level_col].isin(sel_levels) & df[sector_col].isin(sel_sectors)
if opt_genai == "Usuários":       mask &= df.GenAI_user==1
elif opt_genai == "Não usuários": mask &= df.GenAI_user==0
sub = df[mask]

st.title("💰 Quem Usa GenAI Ganha Mais?")
if sub.empty:
    st.warning("Sem dados para esses filtros."); st.stop()

# ------------------- métricas (ordem: min, média, mediana, max)
mini  = sub.salary_mid.min()
mean  = sub.salary_mid.mean()
med   = sub.salary_mid.median()
maxi  = sub.salary_mid.max()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Mínimo",  f"R$ {mini:,.0f}")
c2.metric("Média",   f"R$ {mean:,.0f}")
c3.metric("Mediana", f"R$ {med:,.0f}")
c4.metric("Máximo",  f"R$ {maxi:,.0f}")

# ------------------- distribuição
fig = px.violin(sub,y="salary_mid",x="GenAI_user",box=True,points="all",
                labels={"GenAI_user":"Usa GenAI (0=Não,1=Sim)","salary_mid":"Salário (R$)"},
                title="Distribuição salarial")
st.plotly_chart(fig, use_container_width=True)

g1 = sub[sub.GenAI_user==1].salary_mid
g0 = sub[sub.GenAI_user==0].salary_mid
if len(g1)>10 and len(g0)>10:
    _, p = mannwhitneyu(g1,g0); st.caption(f"Mann-Whitney p = {p:.2e}")

@st.cache_data
def odds(df_):
    m=df_[["salary_mid","GenAI_user",level_col]].dropna()
    if m.GenAI_user.nunique()<2: return None
    m["high"]=(m.salary_mid>m.salary_mid.median()).astype(int)
    X,y = m[["GenAI_user",level_col]], m.high
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("lvl",OneHotEncoder(handle_unknown="ignore"),[level_col]),
            ("num","passthrough",["GenAI_user"])
        ])), ("clf", LogisticRegression(max_iter=1000))
    ]).fit(X,y)
    return float(np.exp(pipe.named_steps["clf"].coef_[0][-1]))
or_val = odds(sub)
st.write(f"**OR(GenAI_user): {or_val:.2f}×** prob. acima da média" if or_val else
         "Sem variação GenAI neste recorte")

# ------------------- prêmio % por setor
st.header("🔍 Prêmio salarial (%) por setor")
agg = (sub.groupby([sector_col,"GenAI_user"]).salary_mid.mean()
          .unstack("GenAI_user")
          .rename(columns={0:"Sem GenAI",1:"Com GenAI"})).dropna()
agg["Prêmio %"] = (agg["Com GenAI"] / agg["Sem GenAI"] - 1) * 100
agg = agg.sort_values("Prêmio %", ascending=False)

fig2 = px.bar(agg, y=agg.index, x="Prêmio %", orientation="h",
              text="Prêmio %", height=650,
              labels={"Prêmio %":"Prêmio %","index":"Setor"},
              title="Prêmio salarial de usar GenAI em cada setor")
fig2.update_traces(texttemplate="%{text:.1f} %")
fig2.update_layout(yaxis={"categoryorder":"total ascending"})
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(
    agg[["Sem GenAI","Com GenAI","Prêmio %"]]
      .style.format({"Sem GenAI":"R$ {:,.0f}","Com GenAI":"R$ {:,.0f}","Prêmio %":"{:+.1f} %"}),
    use_container_width=True
)

st.caption("LBPTech © 2025 · Dados: Data Hackers — demo.")
