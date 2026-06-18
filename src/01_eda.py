"""
01_eda.py

Gera análise exploratória dos dados de treino.

Como rodar:
python src/01_eda.py
"""

import pandas as pd
from pathlib import Path

from config import TRAIN_PATH, REPORTS_DIR, FEATURES, TARGET, LEAKAGE_COLUMNS
from preprocessing import read_csv_safe, select_xy


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Lendo dados de treino...")
    df = read_csv_safe(TRAIN_PATH)

    print(f"Formato do treino: {df.shape[0]} linhas e {df.shape[1]} colunas")

    X, y = select_xy(df)
    selected = X.copy()
    selected[TARGET] = y

    # Salva resumo estatístico simples.
    describe_path = REPORTS_DIR / "describe_selected_features.csv"
    selected.describe(include="all").transpose().to_csv(describe_path, encoding="utf-8-sig")
    print(f"Resumo estatístico salvo em: {describe_path}")

    # Missing por coluna.
    missing = selected.isna().mean().sort_values(ascending=False).reset_index()
    missing.columns = ["coluna", "percentual_missing"]
    missing_path = REPORTS_DIR / "missing_selected_features.csv"
    missing.to_csv(missing_path, index=False, encoding="utf-8-sig")
    print(f"Relatório de missings salvo em: {missing_path}")

    # Distribuição do alvo.
    target_dist = selected[TARGET].value_counts(normalize=False).reset_index()
    target_dist.columns = [TARGET, "n"]
    target_dist["percentual"] = target_dist["n"] / target_dist["n"].sum()
    target_path = REPORTS_DIR / "target_distribution.csv"
    target_dist.to_csv(target_path, index=False, encoding="utf-8-sig")
    print(f"Distribuição do alvo salva em: {target_path}")

    # Colunas removidas por leakage.
    leakage_report = pd.DataFrame({"coluna_removida_por_leakage": LEAKAGE_COLUMNS})
    leakage_path = REPORTS_DIR / "data_leakage_columns.csv"
    leakage_report.to_csv(leakage_path, index=False, encoding="utf-8-sig")
    print(f"Lista de leakage salva em: {leakage_path}")

    # AutoEDA opcional.
    try:
        from ydata_profiling import ProfileReport

        print("Gerando AutoEDA. Pode demorar alguns minutos...")
        sample = selected.sample(min(30000, len(selected)), random_state=42)
        profile = ProfileReport(
            sample,
            title="AutoEDA - Tuberculose - Predição de Abandono",
            explorative=True,
            minimal=True,
        )
        eda_path = REPORTS_DIR / "autoeda_tuberculose.html"
        profile.to_file(eda_path)
        print(f"AutoEDA salvo em: {eda_path}")
    except Exception as e:
        print("Não foi possível gerar o AutoEDA automático.")
        print("O restante da EDA foi salvo normalmente.")
        print(f"Erro: {e}")


if __name__ == "__main__":
    main()
