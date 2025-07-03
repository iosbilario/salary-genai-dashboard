
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

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "df_survey_2024.csv"
KAGGLE_DS = "datahackers/state-of-data-brazil-20242025"
KAGGLE_FILE = "df_survey_2024.csv"

def download_dataset():
    if DATA_PATH.exists():
        return
    st.info("Downloading dataset from Kaggle…")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(KAGGLE_DS, file_name=KAGGLE_FILE, path=str(DATA_DIR))
    zip_path = DATA_DIR / f"{KAGGLE_FILE}.zip"
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(str(DATA_DIR))
    zip_path.unlink()

def faixa_to_midpoint(txt):
    if pd.isna(txt): return np.nan
    nums = [int(x.replace('.', '')) for x in re.findall(r'R\$\s?([0-9.]+)', txt)]
    if 'Mais de' in txt and nums: return nums[0]*1.2
    if 'Menos de' in txt and nums: return nums[0]*0.5
    return sum(nums)/2 if len(nums)==2 else np.nan

@st.cache_data(show_spinner="🔄 loading…")
def load():
    download_dataset()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df['salary_mid'] = df['2.h_faixa_salarial'].apply(faixa_to_midpoint)
    genai_cols = [c for c in df.columns if re.search(r'LLM|IA Generativa', c, re.I)]
    df['GenAI_user'] = df[genai_cols].fillna('').apply(
        lambda r: int(any(str(x).strip().lower() not in ('', 'não', 'nao') for x in r)), axis=1
    )
    return df.dropna(subset=['salary_mid'])

df = load()
st.set_page_config(page_title="Salary × GenAI", layout="wide")
st.title("💰 Quem Usa GenAI Ganha Mais?")

# sidebar
st.sidebar.title("🎛️ Filtros")
levels = sorted(df['2.g_nivel'].dropna().unique())
sectors = sorted(df['2.b_setor'].dropna().unique())
sel_levels = st.sidebar.multiselect("Senioridade", levels, default=levels)
sel_sectors = st.sidebar.multiselect("Setor", sectors, default=sectors)
opt = st.sidebar.radio("GenAI", ["Todos","Usuários","Não usuários"])

mask = df['2.g_nivel'].isin(sel_levels) & df['2.b_setor'].isin(sel_sectors)
if opt=="Usuários":
    mask &= df['GenAI_user']==1
elif opt=="Não usuários":
    mask &= df['GenAI_user']==0
sub = df[mask]

# metrics
col1,col2=st.columns(2)
col1.metric("Mediana", f"R$ {sub['salary_mid'].median():,.0f}")
col2.metric("Média", f"R$ {sub['salary_mid'].mean():,.0f}")

# violin
fig=px.violin(sub,y='salary_mid',x='GenAI_user',box=True,points='all',
              labels={'GenAI_user':'Usa GenAI (0=Não,1=Sim)','salary_mid':'Salário (R$)'},
              title='Distribuição salarial')
st.plotly_chart(fig, use_container_width=True)

# stats
g1=sub[sub.GenAI_user==1]['salary_mid']; g0=sub[sub.GenAI_user==0]['salary_mid']
if len(g1)>10 and len(g0)>10:
    _,p=mannwhitneyu(g1,g0,alternative='two-sided')
    st.caption(f"Mann‑Whitney p={p:.2e}")

@st.cache_data
def odds(data):
    m=data[['salary_mid','GenAI_user','2.g_nivel']].dropna()
    m['high']=(m.salary_mid>m.salary_mid.median()).astype(int)
    X,y=m[['GenAI_user','2.g_nivel']],m['high']
    pipe=Pipeline([('prep',ColumnTransformer([
        ('lvl',OneHotEncoder(handle_unknown='ignore'),['2.g_nivel']),
        ('num','passthrough',['GenAI_user'])
    ])),('clf',LogisticRegression(max_iter=1000))]).fit(X,y)
    return float(np.exp(pipe.named_steps['clf'].coef_[0][-1]))
st.write(f"**OR(GenAI_user): {odds(sub):.2f}×** probabilidade de estar acima da mediana")

st.caption("Dados © Data Hackers — licença MIT. App demo, sem aconselhamento financeiro.")
