import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_data(
    input_path: str,
    output_dir: str,
    target_column: str = "ObesityCategory",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Fungsi untuk melakukan preprocessing data secara otomatis
    dan menyimpan dataset yang siap dilatih.
    """

    # ========================
    # 1. Load Dataset
    # ========================
    data = pd.read_csv(input_path)

    # ========================
    # 2. Encoding Kategorikal
    # ========================
    cat_columns = data.select_dtypes(include='object').columns
    cat_columns = cat_columns.drop(target_column, errors='ignore')

    encoder = LabelEncoder()
    for col in cat_columns:
        data[col] = encoder.fit_transform(data[col])

    # ========================
    # 3. Normalisasi Numerik
    # ========================
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # ========================
    # 4. Pisahkan X dan y
    # ========================
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode target
    y = LabelEncoder().fit_transform(y)

    # ========================
    # 5. Train-Test Split
    # ========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ========================
    # 6. Simpan Dataset
    # ========================
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    pd.DataFrame(y_train, columns=[target_column]).to_csv(
        f"{output_dir}/y_train.csv", index=False
    )
    pd.DataFrame(y_test, columns=[target_column]).to_csv(
        f"{output_dir}/y_test.csv", index=False
    )

    print("Preprocessing selesai. Dataset siap dilatih.")


if __name__ == "__main__":
    preprocess_data(
        input_path="F:\\Kuliah\\Semester 5 (asah)\\SMSML_Vincenz-Reynard-Citro\\dataset_raw\\obesity_data.csv",
        output_dir="F:\\Kuliah\\Semester 5 (asah)\\SMSML_Vincenz-Reynard-Citro\\preprocessing\\dataset_preprocessing"
    )
