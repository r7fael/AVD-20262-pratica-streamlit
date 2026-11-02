# Justificativa das métricas — Dashboard 2 (KPIs)

**Aluno:** Rafael Padilha
**Dataset:** Heart Failure Prediction (fedesoriano, Kaggle) — 917 pacientes após o tratamento da EDA

**Público-alvo geral do dashboard:** equipe clínica e gestão de um serviço de cardiologia (triagem e protocolo de exames).

Todas as métricas são recalculadas com os filtros da sidebar (sexo, faixa etária, intervalos das variáveis). O delta de cada cartão compara o recorte filtrado com uma referência, indicada embaixo do número.

---

## 1. Pacientes

- **Variável:** contagem de linhas do `df_filtrado`
- **Tipo de métrica:** contagem
- **O que mede:** quantos pacientes estão no recorte escolhido nos filtros.
- **Por que foi escolhida:** dá o tamanho da amostra por trás de todas as outras métricas. Uma prevalência de 80% sobre 10 pacientes não vale o mesmo que sobre 500.
- **Público-alvo:** qualquer pessoa que lê o dashboard.
- **Qual decisão pode apoiar:** se o recorte tem pacientes suficientes para tirar uma conclusão ou se é preciso afrouxar os filtros.
- **O que comunica:** "estes números falam de N pacientes" (delta: diferença para a base completa).

## 2. Prevalência de doença cardíaca

- **Variável:** `HeartDisease` (0/1) → `mean() * 100`
- **Tipo de métrica:** proporção
- **O que mede:** percentual de pacientes com diagnóstico positivo no recorte.
- **Por que foi escolhida:** é o desfecho central do dataset. Na base completa é 55,3%, e a EDA mostrou que muda muito entre subgrupos (25,9% nas mulheres contra 63,1% nos homens; 31,1% abaixo de 45 anos contra 71,0% entre 55 e 64).
- **Público-alvo:** gestão e equipe clínica.
- **Qual decisão pode apoiar:** quais perfis de paciente priorizar na triagem e onde investir em rastreamento.
- **O que comunica:** quanto o risco do grupo filtrado fica acima ou abaixo da média (delta em pontos percentuais; cor invertida, porque subir é ruim).

## 3. Doença entre assintomáticos (ASY)

- **Variável:** `HeartDisease` entre os pacientes com `ChestPainType == "ASY"`
- **Tipo de métrica:** taxa condicional
- **O que mede:** percentual de doença entre os pacientes **sem** dor no peito característica.
- **Por que foi escolhida:** foi um dos achados mais contraintuitivos da EDA: 79% dos ASY têm doença, contra 13,9% dos ATA. É o grupo que passa despercebido no exame clínico, porque a queixa não "parece" cardíaca.
- **Público-alvo:** equipe clínica (triagem / pronto atendimento).
- **Qual decisão pode apoiar:** não liberar paciente só porque ele não relata dor típica; pedir teste de esforço/ECG para esse grupo.
- **O que comunica:** "ausência de sintoma não é ausência de doença" (delta: diferença para a prevalência geral do recorte).

## 4. FC máxima mediana dos cardíacos

- **Variável:** `MaxHR`, mediana do grupo Cardíaco, comparada com a do grupo Normal
- **Tipo de métrica:** mediana + variação entre grupos
- **O que mede:** a frequência cardíaca máxima atingida no teste de esforço pelos pacientes com doença.
- **Por que foi escolhida:** junto com `Oldpeak`, é a quantitativa que mais separa as classes (150 bpm no Normal contra 126 no Cardíaco; Mann-Whitney p ≈ 10⁻³⁴). Usei a mediana e não a média porque nenhuma quantitativa passou no Shapiro-Wilk.
- **Público-alvo:** equipe clínica (cardiologia / ergometria).
- **Qual decisão pode apoiar:** usar a FC máxima no esforço como sinal de alerta, levando em conta a idade (FC máxima cai com a idade, r = −0,37).
- **O que comunica:** quantos bpm a menos o paciente cardíaco atinge em relação ao normal no mesmo recorte.

## 5. Sem exame de colesterol

- **Variável:** `Cholesterol_imputado` (True/False) → `sum()` e `%`
- **Tipo de métrica:** taxa (qualidade do dado)
- **O que mede:** percentual de pacientes cujo colesterol veio como 0 (não medido) e foi imputado pela mediana (237 mg/dl).
- **Por que foi escolhida:** a EDA mostrou que a ausência **não é aleatória**: 88,4% dos pacientes sem o exame têm doença, contra 47,7% dos que têm (Qui² p ≈ 10⁻²²). Também é desigual por sexo (22,1% dos homens contra 5,7% das mulheres). Qualquer leitura sobre colesterol depende de saber quanto dele é real.
- **Público-alvo:** gestão do serviço e quem for usar a base (análise de dados / modelagem).
- **Qual decisão pode apoiar:** revisar o protocolo de pedido de exames e desconfiar de conclusões sobre colesterol em recortes com taxa alta de imputação (o toggle "Só colesterol medido" no Dashboard 1 existe por isso).
- **O que comunica:** quanto do colesterol do recorte é medição de verdade.

---

## Visualizações de apoio

- **Pacientes por faixa etária (barras empilhadas por classe):** discretização da `Age` em `<45`, `45-54`, `55-64` e `65+`, como foi feito na EDA (seção de fairness). Mostra volume e composição ao mesmo tempo.
- **% com doença por inclinação do ST, tipo de dor no peito e sexo (barras):** as qualitativas com associação mais forte com o desfecho na EDA.
- Os quatro gráficos são feitos em **Altair com seleção cruzada**: clicar numa barra filtra os outros três, no estilo do Power BI.
- **Tabela Normal × Cardíaco (Mann-Whitney):** medianas por classe e p-valor, para não confundir significância estatística com relevância prática.
