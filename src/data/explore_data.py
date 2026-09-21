
from pathlib import Path

import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / "raw" / "students.csv"

    data = pd.read_csv(file_path)

    print("=" * 50)
    print("EDUADAPT AI - DATA EXPLORATION")
    print("=" * 50)

    print("\nShape:")
    print(data.shape)

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nDuplicate rows:")
    print(data.duplicated().sum())

    print("\nStatistik numerik:")
    print(data.describe())

    print("\nDistribusi minat:")
    print(data["interest"].value_counts())


if __name__ == "__main__":
    main()