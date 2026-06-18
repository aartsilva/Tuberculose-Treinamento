# Documentação das Decisões de Manipulação dos Dados

## 1. Variável alvo

A variável alvo é `ltfu`, que representa abandono do tratamento.

## 2. Seleção de preditores

Foram selecionadas variáveis disponíveis antes ou no início do acompanhamento do paciente, incluindo idade, sexo, escolaridade, raça/cor, forma clínica, exames iniciais, agravos e populações especiais.

## 3. Remoção de vazamento de dados

Foram removidas variáveis relacionadas ao encerramento ou ao acompanhamento posterior, como `SITUA_ENCE` e `DT_ENCERRA`.

Essas variáveis não devem ser usadas porque só são conhecidas depois que o desfecho já ocorreu.

## 4. Tratamento de valores ausentes

- Numéricas: imputação pela mediana.
- Categóricas: imputação pela moda.

## 5. Escalonamento

As variáveis numéricas são transformadas com `RobustScaler`, pois ele é menos sensível a outliers.

## 6. Codificação de categóricas

As variáveis categóricas são codificadas com `OneHotEncoder(handle_unknown="ignore")`.

Isso permite que o modelo receba categorias novas nos dados de teste sem quebrar.

## 7. Modelos

Foram treinados dois modelos:

1. Regressão logística: baseline interpretável.
2. Rede neural: modelo não linear exigido no trabalho.

## 8. Avaliação

Foram usadas:

- acurácia;
- precisão;
- recall;
- F1-score;
- ROC-AUC.

O F1-score foi usado como métrica principal porque equilibra precisão e recall.
