import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from src.data_loader import FAIXAS_ETARIAS, load_data

st.set_page_config(
    page_title="Heart Failure Prediction · EDA",
    page_icon=":material/cardiology:",
    layout="wide",
)

df = load_data()

# Variáveis quantitativas do estudo (mesma lista VARS_QUANT do notebook de EDA)
VARS_QUANT = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
ROTULOS = {
    "Age": "Idade (anos)",
    "RestingBP": "Pressão em repouso (mm Hg)",
    "Cholesterol": "Colesterol (mg/dl)",
    "MaxHR": "FC máxima (bpm)",
    "Oldpeak": "Oldpeak (depressão ST)",
}

# Paleta da EDA: rosa para Normal, roxo para Cardíaco (tons ajustados para ler bem no modo claro e escuro)
CORES_CLASSE = {"Normal": "#d99a8c", "Cardíaco": "#5b3f8c"}
ORDEM_CLASSE = ["Normal", "Cardíaco"]

# Sidebar: filtros
with st.sidebar:
    st.header(":material/tune: Filtros")

    sexos = st.pills(
        "Sexo",
        options=["Feminino", "Masculino"],
        default=["Feminino", "Masculino"],
        selection_mode="multi",
    )
    faixas = st.pills(
        "Faixa etária",
        options=FAIXAS_ETARIAS,
        default=FAIXAS_ETARIAS,
        selection_mode="multi",
    )

    st.divider()

    variaveis = st.multiselect(
        "Variáveis quantitativas",
        options=VARS_QUANT,
        default=VARS_QUANT,
        format_func=ROTULOS.get,
        help="Variáveis usadas na matriz de correlação e disponíveis no histograma.",
    )

    separar_por_classe = st.toggle(
        "Distinguir por classe",
        value=True,
        help="Colore os gráficos pela classe (doenca_cardiaca: Normal ou Cardíaco).",
    )

    so_colesterol_medido = st.toggle(
        "Só colesterol medido",
        value=False,
        help=(
            "Ignora os 171 valores de colesterol imputados pela mediana (237 mg/dl). "
            "Na EDA, a ausência do exame estava associada ao desfecho (88,4% contra 47,7% de doença)."
        ),
    )

    # Um slider de intervalo para cada variável selecionada
    intervalos = {}
    with st.expander("Intervalos das variáveis", icon=":material/straighten:"):
        for var in variaveis:
            if var == "Oldpeak":
                minimo, maximo, passo = float(df[var].min()), float(df[var].max()), 0.1
            else:
                minimo, maximo, passo = int(df[var].min()), int(df[var].max()), 1
            intervalos[var] = st.slider(
                ROTULOS[var],
                min_value=minimo,
                max_value=maximo,
                value=(minimo, maximo),
                step=passo,
                key=f"intervalo_{var}",
            )

# Aplica os filtros
df_filtrado = df[df["sexo_label"].isin(sexos) & df["faixa_etaria"].isin(faixas)].copy()
if so_colesterol_medido:
    df_filtrado.loc[df_filtrado["Cholesterol_imputado"], "Cholesterol"] = np.nan
for var, (inicio, fim) in intervalos.items():
    # Mantém os NaN (colesterol imputado oculto) para não remover o paciente das outras análises
    df_filtrado = df_filtrado[df_filtrado[var].between(inicio, fim) | df_filtrado[var].isna()]

# Cabeçalho
st.title(":material/cardiology: Heart Failure Prediction")
st.caption(
    "EDA de Rafael Padilha · dataset *Heart Failure Prediction* (fedesoriano, Kaggle) · "
    f"**{len(df_filtrado)}** de **{len(df)}** pacientes dentro dos filtros"
)

if df_filtrado.empty:
    st.warning("Nenhum paciente dentro dos filtros escolhidos. Ajuste a sidebar.", icon=":material/filter_alt_off:")
    st.stop()

aba_d1, aba_d2 = st.tabs([":material/scatter_plot: Correlação e distribuição", ":material/monitoring: KPIs"])

# Dashboard 1: correlação + distribuição das variáveis quantitativas
with aba_d1:
    col_correlacao, col_distribuicao = st.columns([1, 1.4])

    with col_correlacao, st.container(border=True, height="stretch"):
        st.subheader("Correlação")
        metodo = st.segmented_control(
            "Método",
            options=["spearman", "pearson", "kendall"],
            default="spearman",
            format_func=str.capitalize,
            help="Nenhuma variável passou no Shapiro-Wilk, por isso Spearman é o padrão.",
        )

        if len(variaveis) < 2:
            st.warning("Selecione pelo menos 2 variáveis para calcular a correlação.")
        else:
            corr = df_filtrado[variaveis].corr(method=metodo or "spearman")
            # Esconde o triângulo superior (espelho do inferior), como no notebook
            corr_tri = corr.where(np.tril(np.ones(corr.shape, dtype=bool)))
            fig = px.imshow(
                corr_tri,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_colorbar=dict(title="r"),
                xaxis_title=None,
                yaxis_title=None,
            )
            st.plotly_chart(fig, alt=f"Matriz de correlação ({metodo}) entre as variáveis quantitativas")

            # Par mais forte em módulo
            pares = corr_tri.where(~np.eye(len(corr), dtype=bool)).stack()
            var_a, var_b = pares.abs().idxmax()
            r = pares[(var_a, var_b)]
            forca = "fraca" if abs(r) < 0.3 else "moderada" if abs(r) < 0.7 else "forte"
            st.caption(
                f"Par mais forte: **{var_a} × {var_b}** (r = {r:.2f}, {forca}). "
                "|r| < 0,3 fraca · 0,3 a 0,7 moderada · > 0,7 forte."
            )

    with col_distribuicao, st.container(border=True, height="stretch"):
        st.subheader("Distribuição")
        opcoes_hist = variaveis or VARS_QUANT
        var_hist = st.selectbox(
            "Variável do histograma",
            options=opcoes_hist,
            index=opcoes_hist.index("MaxHR") if "MaxHR" in opcoes_hist else 0,
            format_func=ROTULOS.get,
        )

        fig = px.histogram(
            df_filtrado.dropna(subset=[var_hist]),
            x=var_hist,
            color="doenca_cardiaca" if separar_por_classe else None,
            color_discrete_map=CORES_CLASSE,
            category_orders={"doenca_cardiaca": ORDEM_CLASSE},
            color_discrete_sequence=["#8f7ab3"],
            barmode="overlay",
            opacity=0.75,
            marginal="box",
            nbins=30,
            labels={var_hist: ROTULOS[var_hist], "doenca_cardiaca": "Classe"},
        )
        mediana = df_filtrado[var_hist].median()
        fig.add_vline(x=mediana, line_dash="dash", line_width=1.5, annotation_text=f"mediana {mediana:.1f}")
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Pacientes",
            legend=dict(orientation="h", y=1.12, x=0, title=None),
        )
        st.plotly_chart(fig, alt=f"Histograma de {ROTULOS[var_hist]}")

        serie = df_filtrado[var_hist].dropna()
        with st.container(horizontal=True):
            st.metric("Média", f"{serie.mean():.1f}", border=True)
            st.metric("Mediana", f"{serie.median():.1f}", border=True)
            st.metric("Desvio padrão", f"{serie.std():.1f}", border=True)
            st.metric("Skewness", f"{serie.skew():.2f}", border=True)

    with st.container(border=True):
        st.subheader("Dispersão entre duas quantitativas")
        st.caption("Idade × FC máxima é o par com a correlação mais forte da EDA (−0,37) e tem explicação fisiológica.")
        col_x, col_y = st.columns(2)
        eixo_x = col_x.selectbox("Eixo X", VARS_QUANT, index=0, format_func=ROTULOS.get)
        eixo_y = col_y.selectbox("Eixo Y", VARS_QUANT, index=3, format_func=ROTULOS.get)
        fig = px.scatter(
            df_filtrado.dropna(subset=[eixo_x, eixo_y]),
            x=eixo_x,
            y=eixo_y,
            color="doenca_cardiaca" if separar_por_classe else None,
            color_discrete_map=CORES_CLASSE,
            category_orders={"doenca_cardiaca": ORDEM_CLASSE},
            color_discrete_sequence=["#8f7ab3"],
            opacity=0.75,
            labels={**ROTULOS, "doenca_cardiaca": "Classe"},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1, x=0, title=None))
        st.plotly_chart(fig, alt=f"Dispersão de {ROTULOS[eixo_x]} por {ROTULOS[eixo_y]}")


def taxa_doenca(dados, coluna, ordem=None):
    """% de pacientes com diagnóstico positivo em cada categoria da coluna."""
    taxa = (
        dados.groupby(coluna, observed=True)["HeartDisease"]
        .agg(pct="mean", n="size")
        .reset_index()
    )
    taxa["pct"] = taxa["pct"] * 100
    if ordem:
        taxa[coluna] = pd.Categorical(taxa[coluna], categories=ordem, ordered=True)
        taxa = taxa.sort_values(coluna)
    return taxa


def grafico_taxa(dados, coluna, titulo, descricao, ordem=None):
    taxa = taxa_doenca(dados, coluna, ordem)
    fig = px.bar(
        taxa,
        x=coluna,
        y="pct",
        text=taxa["pct"].map(lambda v: f"{v:.0f}%"),
        hover_data={"n": True, "pct": ":.1f"},
        color_discrete_sequence=[CORES_CLASSE["Cardíaco"]],
        labels={"pct": "% com doença", coluna: "", "n": "Pacientes"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), yaxis_range=[0, 105])
    with st.container(border=True):
        st.markdown(f"**{titulo}**")
        st.caption(descricao)
        st.plotly_chart(fig, alt=f"Percentual com doença cardíaca por {titulo.lower()}", height=280)


# Dashboard 2: KPIs
with aba_d2:
    pct_base = df["HeartDisease"].mean() * 100
    pct_filtro = df_filtrado["HeartDisease"].mean() * 100
    n_imputados = int(df_filtrado["Cholesterol_imputado"].sum())

    with st.container(horizontal=True):
        st.metric(
            "Pacientes",
            f"{len(df_filtrado)}",
            len(df_filtrado) - len(df),
            delta_color="off",
            delta_description="vs. base completa",
            border=True,
        )
        st.metric(
            "Com doença cardíaca",
            f"{pct_filtro:.1f}%",
            f"{pct_filtro - pct_base:+.1f} p.p.",
            delta_color="inverse",
            delta_description="vs. base completa",
            border=True,
        )
        st.metric(
            "Idade mediana",
            f"{df_filtrado['Age'].median():.0f} anos",
            f"{df_filtrado['Age'].median() - df['Age'].median():+.0f}",
            delta_color="off",
            delta_description="vs. base completa",
            border=True,
        )
        st.metric(
            "FC máxima mediana",
            f"{df_filtrado['MaxHR'].median():.0f} bpm",
            f"{df_filtrado['MaxHR'].median() - df['MaxHR'].median():+.0f}",
            delta_color="off",
            delta_description="vs. base completa",
            border=True,
        )
        st.metric(
            "Colesterol imputado",
            f"{100 * n_imputados / len(df_filtrado):.1f}%",
            f"{n_imputados} pacientes",
            delta_color="off",
            delta_description="imputados pela mediana",
            border=True,
        )

    linha1 = st.columns(2)
    with linha1[0]:
        grafico_taxa(
            df_filtrado, "ST_Slope", "Inclinação do segmento ST",
            "A variável qualitativa mais informativa da EDA (Qui² p ≈ 8 × 10⁻⁷⁸).",
            ordem=["Up", "Flat", "Down"],
        )
    with linha1[1]:
        grafico_taxa(
            df_filtrado, "ChestPainType", "Tipo de dor no peito",
            "ASY (assintomático) concentra mais doença: sem dor característica não significa sem doença.",
            ordem=["ASY", "NAP", "ATA", "TA"],
        )

    linha2 = st.columns(2)
    with linha2[0]:
        grafico_taxa(
            df_filtrado, "sexo_label", "Sexo",
            "Diferença de quase 40 p.p. — pode ser risco real ou viés de encaminhamento.",
        )
    with linha2[1]:
        grafico_taxa(
            df_filtrado, "faixa_etaria", "Faixa etária",
            "O risco cresce até os 64 anos e depois estabiliza.",
            ordem=FAIXAS_ETARIAS,
        )

    with st.container(border=True):
        st.markdown("**Normal × Cardíaco: medianas e teste de Mann-Whitney**")
        st.caption(
            "Nenhuma quantitativa é normal, então a comparação usa o teste não-paramétrico. "
            "Significância estatística não é o mesmo que relevância prática: olhe a diferença de mediana junto com o p-valor."
        )
        linhas = []
        for var in VARS_QUANT:
            normal = df_filtrado.loc[df_filtrado["doenca_cardiaca"] == "Normal", var].dropna()
            cardiaco = df_filtrado.loc[df_filtrado["doenca_cardiaca"] == "Cardíaco", var].dropna()
            p_valor = stats.mannwhitneyu(normal, cardiaco).pvalue if len(normal) and len(cardiaco) else np.nan
            linhas.append({
                "Variável": ROTULOS[var],
                "Mediana Normal": normal.median(),
                "Mediana Cardíaco": cardiaco.median(),
                "Diferença": cardiaco.median() - normal.median(),
                "p-valor": p_valor,
                "Significativo (α = 0,05)": bool(p_valor < 0.05),
            })
        st.dataframe(
            pd.DataFrame(linhas),
            hide_index=True,
            column_config={
                "Mediana Normal": st.column_config.NumberColumn(format="%.1f"),
                "Mediana Cardíaco": st.column_config.NumberColumn(format="%.1f"),
                "Diferença": st.column_config.NumberColumn(format="%+.1f"),
                "p-valor": st.column_config.NumberColumn(format="%.2e"),
            },
            alt="Comparação das medianas entre os grupos Normal e Cardíaco",
        )
