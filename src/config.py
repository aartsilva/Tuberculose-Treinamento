"""
Arquivo central de configuração do projeto.

Aqui ficam:
- caminhos dos arquivos;
- variável alvo;
- variáveis preditoras;
- colunas removidas por risco de data leakage.

Data leakage significa usar no treino uma informação que só existe depois
do desfecho do tratamento. Isso faz o modelo parecer bom, mas ele não serve
na prática clínica.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

TRAIN_PATH = DATA_DIR / "treino.csv"
TEST1_PATH = DATA_DIR / "teste1.csv"
TEST2_PATH = DATA_DIR / "teste2.csv"

TARGET = "ltfu"

# Variáveis com risco de vazamento de informação.
# Elas informam encerramento, transferência ou resultados posteriores.
LEAKAGE_COLUMNS = [
    "SITUA_ENCE",
    "DT_ENCERRA",
    "SITUA_9_M",
    "SITUA_12_M",
    "TRANSF",
    "UF_TRANSF",
    "MUN_TRANSF",
    "BAC_APOS_6",
]

# Variáveis escolhidas por relevância aparente e disponibilidade antes/durante início do tratamento.
# Mantivemos aproximadamente 25 preditores, como sugerido no roteiro.
NUMERIC_FEATURES = [
    "idade_anos",
    "NU_ANO",
    "NU_CONTATO",
]

CATEGORICAL_FEATURES = [
    "SG_UF_NOT",
    "CS_SEXO",
    "CS_GESTANT",
    "CS_RACA",
    "CS_ESCOL_N",
    "TRATAMENTO",
    "INSTITUCIO",
    "RAIOX_TORA",
    "TESTE_TUBE",
    "FORMA",
    "AGRAVAIDS",
    "AGRAVALCOO",
    "AGRAVDIABE",
    "AGRAVDOENC",
    "AGRAVOUTRA",
    "BACILOSC_E",
    "BACILOS_E2",
    "BACILOSC_O",
    "CULTURA_ES",
    "CULTURA_OU",
    "HIV",
    "HISTOPATOL",
    "TRAT_SUPER",
    "DOENCA_TRA",
    "POP_LIBER",
    "POP_RUA",
    "POP_SAUDE",
    "POP_IMIG",
    "BENEF_GOV",
    "AGRAVDROGA",
    "AGRAVTABAC",
    "TEST_MOLEC",
    "TEST_SENSI",
    "ANT_RETRO",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
