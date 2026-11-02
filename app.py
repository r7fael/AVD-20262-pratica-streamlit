import altair as alt
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

# Paleta da EDA: coral para Normal, roxo para Cardíaco (contraste ≥ 3:1 no modo claro e no escuro)
CORES_CLASSE = {"Normal": "#c4786a", "Cardíaco": "#7f5fb8"}
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


def mediana(dados, var, classe=None):
    if classe:
        dados = dados[dados["doenca_cardiaca"] == classe]
    return dados[var].median()


def grafico_taxa(dados, coluna, titulo, ordem, selecao, outras_selecoes):
    """% com doença por categoria. O clique seleciona; os filtros vêm das seleções dos outros gráficos."""
    base = alt.Chart(dados, title=titulo)
    for outra in outras_selecoes:
        base = base.transform_filter(outra)
    base = base.transform_aggregate(pct="mean(HeartDisease)", n="count()", groupby=[coluna])
    eixo_x = alt.X(f"{coluna}:N", sort=ordem, title=None, axis=alt.Axis(labelAngle=0))
    barras = (
        base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=eixo_x,
            y=alt.Y("pct:Q", title="% com doença", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.value(CORES_CLASSE["Cardíaco"]),
            opacity=alt.condition(selecao, alt.value(1), alt.value(0.35)),
            tooltip=[
                alt.Tooltip(f"{coluna}:N", title="Categoria"),
                alt.Tooltip("pct:Q", title="% com doença", format=".1%"),
                alt.Tooltip("n:Q", title="Pacientes"),
            ],
        )
        .add_params(selecao)
    )
    rotulos = base.mark_text(dy=-8, fontWeight=600).encode(
        x=eixo_x, y="pct:Q", text=alt.Text("pct:Q", format=".0%")
    )
    return (barras + rotulos).properties(width=330, height=220)


# Dashboard 2: KPIs
with aba_d2:
    prevalencia_base = df["HeartDisease"].mean() * 100
    prevalencia = df_filtrado["HeartDisease"].mean() * 100
    asy = df_filtrado[df_filtrado["ChestPainType"] == "ASY"]
    prevalencia_asy = asy["HeartDisease"].mean() * 100 if len(asy) else np.nan
    fc_cardiaco = mediana(df_filtrado, "MaxHR", "Cardíaco")
    fc_normal = mediana(df_filtrado, "MaxHR", "Normal")
    n_sem_colesterol = int(df_filtrado["Cholesterol_imputado"].sum())

    with st.container(horizontal=True):
        st.metric(
            "Pacientes",
            f"{len(df_filtrado)}",
            len(df_filtrado) - len(df),
            delta_color="off",
            delta_description="vs. base completa",
            help="Contagem de pacientes dentro dos filtros da sidebar.",
            border=True,
        )
        st.metric(
            "Prevalência de doença cardíaca",
            f"{prevalencia:.1f}%",
            round(prevalencia - prevalencia_base, 1) and f"{prevalencia - prevalencia_base:+.1f} p.p.",
            delta_color="inverse",
            delta_description="vs. base completa",
            help="Proporção de pacientes com diagnóstico positivo (HeartDisease = 1).",
            border=True,
        )
        st.metric(
            "Doença entre assintomáticos",
            "—" if np.isnan(prevalencia_asy) else f"{prevalencia_asy:.1f}%",
            None if np.isnan(prevalencia_asy) else f"{prevalencia_asy - prevalencia:+.1f} p.p.",
            delta_color="inverse",
            delta_description="vs. prevalência geral",
            help="% com doença entre os pacientes sem dor no peito característica (ChestPainType = ASY).",
            border=True,
        )
        st.metric(
            "FC máxima mediana dos cardíacos",
            "—" if np.isnan(fc_cardiaco) else f"{fc_cardiaco:.0f} bpm",
            None if np.isnan(fc_cardiaco) or np.isnan(fc_normal) else f"{fc_cardiaco - fc_normal:+.0f} bpm",
            delta_color="off",
            delta_description="vs. pacientes normais",
            help="Mediana da frequência cardíaca máxima no esforço do grupo Cardíaco, comparada à do grupo Normal.",
            border=True,
        )
        st.metric(
            "Sem exame de colesterol",
            f"{100 * n_sem_colesterol / len(df_filtrado):.1f}%",
            f"{n_sem_colesterol} pacientes",
            delta_color="off",
            delta_description="imputados pela mediana",
            help="Taxa de pacientes cujo colesterol veio como 0 (não medido) e foi imputado.",
            border=True,
        )

    with st.container(border=True):
        st.subheader("Onde a doença se concentra")
        st.caption(
            "Clique numa barra para filtrar os outros gráficos (shift + clique seleciona várias; "
            "clique fora das barras para limpar)."
        )

        dados_alt = df_filtrado[
            ["HeartDisease", "doenca_cardiaca", "faixa_etaria", "ST_Slope", "ChestPainType", "sexo_label"]
        ].astype({"faixa_etaria": str})

        sel_faixa = alt.selection_point(fields=["faixa_etaria"], name="faixa")
        sel_st = alt.selection_point(fields=["ST_Slope"], name="st")
        sel_dor = alt.selection_point(fields=["ChestPainType"], name="dor")
        sel_sexo = alt.selection_point(fields=["sexo_label"], name="sexo")
        selecoes = [sel_faixa, sel_st, sel_dor, sel_sexo]

        def outras(selecao):
            return [s for s in selecoes if s is not selecao]

        # Idade discretizada em faixas: contagem de pacientes por classe
        base_faixa = alt.Chart(dados_alt, title="Pacientes por faixa etária")
        for outra in outras(sel_faixa):
            base_faixa = base_faixa.transform_filter(outra)
        grafico_faixa = (
            base_faixa.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("faixa_etaria:N", sort=FAIXAS_ETARIAS, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("count():Q", title="Pacientes"),
                color=alt.Color(
                    "doenca_cardiaca:N",
                    title="Classe",
                    scale=alt.Scale(domain=ORDEM_CLASSE, range=[CORES_CLASSE[c] for c in ORDEM_CLASSE]),
                    legend=alt.Legend(orient="top"),
                ),
                order=alt.Order("doenca_cardiaca:N", sort="descending"),
                opacity=alt.condition(sel_faixa, alt.value(1), alt.value(0.35)),
                tooltip=[
                    alt.Tooltip("faixa_etaria:N", title="Faixa etária"),
                    alt.Tooltip("doenca_cardiaca:N", title="Classe"),
                    alt.Tooltip("count():Q", title="Pacientes"),
                ],
            )
            .add_params(sel_faixa)
            .properties(width=330, height=220)
        )

        painel = alt.vconcat(
            alt.hconcat(
                grafico_faixa,
                grafico_taxa(dados_alt, "ST_Slope", "% com doença por inclinação do ST",
                             ["Up", "Flat", "Down"], sel_st, outras(sel_st)),
            ),
            alt.hconcat(
                grafico_taxa(dados_alt, "ChestPainType", "% com doença por tipo de dor no peito",
                             ["ASY", "NAP", "ATA", "TA"], sel_dor, outras(sel_dor)),
                grafico_taxa(dados_alt, "sexo_label", "% com doença por sexo",
                             ["Feminino", "Masculino"], sel_sexo, outras(sel_sexo)),
            ),
        ).configure_title(anchor="start", fontSize=14)

        st.altair_chart(
            painel,
            alt="Painel interligado: pacientes por faixa etária e % com doença por inclinação do ST, dor no peito e sexo",
        )

        with st.expander("Como ler", icon=":material/lightbulb:"):
            st.markdown(
                "- **ST plano (Flat) ou descendente (Down)** inverte o prognóstico em relação ao **Up**: "
                "é a qualitativa mais associada ao desfecho na EDA (Qui² p ≈ 8 × 10⁻⁷⁸).\n"
                "- **ASY** (assintomático) é a categoria com mais doença: sem dor característica não é sem doença.\n"
                "- A diferença entre **sexos** (≈ 40 p.p.) pode ser risco real ou viés de encaminhamento.\n"
                "- O risco cresce com a **idade** até os 64 anos e depois estabiliza."
            )

    with st.container(border=True):
        st.subheader("Normal × Cardíaco: medianas e teste de Mann-Whitney")
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
