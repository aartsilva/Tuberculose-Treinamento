# Passo a passo absoluto para fazer o trabalho

## Parte A — Preparação no computador

1. Baixe este projeto.
2. Extraia o ZIP.
3. Abra a pasta extraída no VSCode.
4. Copie os arquivos `treino.csv`, `teste1.csv` e `teste2.csv` para a pasta `data`.

A pasta deve ficar assim:

```text
data/treino.csv
data/teste1.csv
data/teste2.csv
```

## Parte B — Ambiente Python

No terminal do VSCode:

```bash
python -m venv .venv
```

Ative:

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Instale:

```bash
pip install -r requirements.txt
```

## Parte C — Rodar por blocos ou código corrido?

Use assim:

- Para aprender e testar: rode por partes.
- Para entregar no GitHub: deixe os scripts corridos da pasta `src`.

A ordem correta é:

```bash
python src/01_eda.py
python src/02_train_logistic.py --sample 120000
python src/03_train_neural_network.py --sample 120000 --epochs 10
python src/04_train_final.py
python src/05_explainability.py
```

Depois, se tudo funcionar, rode sem sample:

```bash
python src/02_train_logistic.py
python src/03_train_neural_network.py --epochs 30
python src/04_train_final.py
python src/05_explainability.py
```

## Parte D — Rodar API

```bash
uvicorn app.api:app --reload
```

Abra:

```text
http://127.0.0.1:8000/docs
```

Clique em `POST /predict`, depois `Try it out`.

## Parte E — O que colocar no relatório

1. Introdução: problema da tuberculose e abandono.
2. Método: dados SINAN/DataSUS, variável alvo `ltfu`, limpeza, pipeline.
3. Modelos: regressão logística e rede neural.
4. Métricas: acurácia, precisão, recall, F1 e ROC-AUC.
5. Resultados: colocar tabela com métricas de `reports/metrics_final_test2.json`.
6. Discussão: interpretar desempenho, limitações e data leakage.
7. Considerações finais: uso como apoio para profissionais da saúde.
