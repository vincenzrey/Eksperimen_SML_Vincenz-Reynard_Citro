import pandas as pd
import os
import argparse
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def preprocess_data(
    input_path: str,
    output_path: str,
    target_column: str = "ObesityCategory"
):
    # 1. Load dataset
    df = pd.read_csv(input_path)

    # 2. Encoding fitur kategorikal (kecuali target)
    categorical_columns = df.select_dtypes(include=["object"]).columns
    categorical_columns = categorical_columns.drop(target_column, errors="ignore")

    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])

    # 3. Scaling fitur numerik (kecuali target)
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_columns = numeric_columns.drop(target_column, errors="ignore")

    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # 4. Simpan dataset hasil preprocessing (BELUM split)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "obesity_preprocessed.csv")
    df.to_csv(output_file, index=False)

    print("Preprocessing selesai.")
    print(f"Dataset siap dilatih disimpan di: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated preprocessing dataset obesity")
    parser.add_argument("--input", type=str, required=True, help="Path dataset raw")
    parser.add_argument("--output", type=str, required=True, help="Folder output preprocessing")

    args = parser.parse_args()

    preprocess_data(
        input_path=args.input,
        output_path=args.output
    )
