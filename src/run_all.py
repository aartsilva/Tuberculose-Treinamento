"""
Roda o fluxo básico do projeto.

Como rodar:
python src/run_all.py

Atenção:
- Este script usa amostra para acelerar os primeiros testes.
- Para o trabalho final, rode os scripts manualmente sem --sample.
"""

import subprocess
import sys


def run(cmd):
    print("\n" + "=" * 80)
    print("Rodando:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main():
    py = sys.executable

    run([py, "src/01_eda.py"])
    run([py, "src/02_train_logistic.py", "--sample", "120000"])
    run([py, "src/03_train_neural_network.py", "--sample", "120000", "--epochs", "10"])
    run([py, "src/04_train_final.py"])
    run([py, "src/05_explainability.py"])

    print("\nTudo pronto!")
    print("Veja os resultados na pasta reports/")
    print("Veja os modelos na pasta models/")


if __name__ == "__main__":
    main()
