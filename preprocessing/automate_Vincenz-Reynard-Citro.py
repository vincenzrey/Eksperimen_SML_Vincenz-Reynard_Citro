import pandas as pd
import os
import argparse
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_data(
    input_path: str,
    output_path: str,
    target_column: str = "ObesityCategory",
    test_size: float = 0.2,
    random_state: int = 42
):
    # 1. Load Dataset
    data = pd.read_csv(input_path)

    # 2. Encoding Kategorikal
    encoder = LabelEncoder()
    categorical_columns = ['Gender', 'ObesityCategory']
    for col in categorical_columns:
        data[col] = encoder.fit_transform(data[col])

    # 3. Normalisasi Numerik
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # 4. Pisahkan X dan y
    X = data.drop(columns=[target_column])
    y = data[target_column]

    y = LabelEncoder().fit_transform(y)

    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 6. Simpan Dataset
    os.makedirs(output_path, exist_ok=True)

    X_train.to_csv(f"{output_path}/X_train.csv", index=False)
    X_test.to_csv(f"{output_path}/X_test.csv", index=False)
    pd.DataFrame(y_train, columns=[target_column]).to_csv(
        f"{output_path}/y_train.csv", index=False
    )
    pd.DataFrame(y_test, columns=[target_column]).to_csv(
        f"{output_path}/y_test.csv", index=False
    )

    print("Preprocessing selesai. Dataset siap dilatih.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated preprocessing")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    preprocess_data(
        input_path=args.input,
        output_path=args.output
    )
