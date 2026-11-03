# Heart Failure Prediction — EDA e dashboards em Streamlit

**App publicado:** <!-- LINK_DO_APP --> _(link do Streamlit Community Cloud, adicionar após o deploy)_

Projeto da disciplina **Análise e Visualização de Dados** (CESAR School, 2026.2), aulas 21 a 23. A partir da minha EDA do dataset *Heart Failure Prediction* (fedesoriano, Kaggle), construí dois dashboards interativos:

- **Dashboard 1 — correlação e distribuição** das variáveis quantitativas (idade, pressão em repouso, colesterol, FC máxima e Oldpeak).
- **Dashboard 2 — KPIs**: 5 métricas e um painel de gráficos com seleção cruzada mostrando onde a doença cardíaca se concentra.

## Por que este projeto é interessante

O dataset parece limpo (nenhum `NaN`), mas guarda exames não realizados como **zero**: 172 pacientes com colesterol 0 e 1 com pressão 0. A EDA mostrou que essa ausência **não é aleatória**: 88,4% dos pacientes sem exame de colesterol têm doença, contra 47,7% dos que têm (Qui² p ≈ 10⁻²²). Por isso o colesterol foi imputado pela mediana com uma flag (`Cholesterol_imputado`), em vez de as linhas serem removidas, e o app tem um filtro "Só colesterol medido" para isolar os valores reais.

Outros achados que o dashboard deixa visíveis:

- **79%** dos pacientes **assintomáticos** (sem dor no peito típica) têm doença.
- A **inclinação do segmento ST** praticamente inverte o prognóstico (Up: 19,7% com doença; Flat: 82,8%).
- Pacientes cardíacos atingem FC máxima mediana **24 bpm menor** no esforço.
- Há **~40 p.p.** de diferença de diagnóstico entre homens e mulheres: pode ser risco real ou viés de encaminhamento.

## Stack

- **Python 3.11–3.15**
- **Streamlit** (app, tabs, `st.metric`, tema em `.streamlit/config.toml`)
- **pandas / NumPy / SciPy** (tratamento e testes estatísticos: Spearman, Mann-Whitney)
- **Plotly** (heatmap, histograma e dispersão do Dashboard 1)
- **Altair** (painel com seleção cruzada do Dashboard 2)
- EDA em **Jupyter** com Matplotlib e Seaborn

## Estrutura

| Caminho | Conteúdo |
| --- | --- |
| `app.py` | Aplicação Streamlit (sidebar de filtros + 2 dashboards em abas) |
| `src/data_loader.py` | Leitura do dataset tratado com `@st.cache_data` |
| `data/clean/dataset.csv` | Dataset tratado (917 pacientes), exportado pelo notebook |
| `notebooks/eda_original_rafaelPadilha.ipynb` | EDA original |
| `notebooks/eda_posterior_rafaelPadilha.ipynb` | EDA com a exportação dos datasets usados pelo app |
| `entregas/` | Rascunhos dos dois dashboards e justificativa das métricas |
| `prompts/` | Templates de prompt e declaração de uso de IA |
| `.streamlit/config.toml` | Identidade visual (paleta da EDA: roxo = Cardíaco, coral = Normal) |

`data/raw/` não é versionado: o dado bruto é baixado do Kaggle pelo notebook.

## Como rodar localmente

```bash
# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
streamlit run app.py
```

## Uso de IA

Declarado em [prompts/declaracao-uso-ia.md](prompts/declaracao-uso-ia.md).
