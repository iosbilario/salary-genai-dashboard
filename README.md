
# Salary × GenAI Dashboard 🇧🇷

Streamlit app to explore the **State of Data Brazil 2024/2025** survey and answer:

> **Do professionals who work with Generative AI really earn more?**

## Quick start

```bash
git clone https://github.com/iosbilario/salary-genai-dashboard.git 
cd salary-genai-dashboard
pip install -r requirements.txt
export KAGGLE_USERNAME=your_user
export KAGGLE_KEY=your_key
streamlit run streamlit_app.py
```

### Deploy on Streamlit Cloud

1. Push/fork this repo to GitHub (public).
2. Go to https://streamlit.io/cloud → *New App*.
3. Set main file to `streamlit_app.py`.
4. Add **Secrets**:

```
KAGGLE_USERNAME = your_user
KAGGLE_KEY = your_key
```

5. Deploy. First cold start downloads the CSV automatically.

### Auto-refresh dataset

A GitHub Action (`.github/workflows/update-data.yml`) runs weekly (Sunday) and on-demand, pulls the latest CSV from Kaggle, commits if changed, and Streamlit Cloud redeploys.

### Structure

```
salary-genai-dashboard/
├── streamlit_app.py
├── requirements.txt
├── README.md
├── data/                 # CSV saved here by app or workflow
└── .github/workflows/
    └── update-data.yml
```

Data © Data Hackers — MIT license. This repo redistributes only the CSV with attribution.
