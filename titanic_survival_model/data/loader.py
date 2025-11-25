import pandas as pd


TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)


def load_raw_data(url=TITANIC_URL):
    """Load the raw Titanic dataset from a URL."""
    return pd.read_csv(url)


def preprocess_titanic(df):
    """
    Preprocess Titanic data:
    - Keep selected columns
    - Convert categorical variables
    - Drop rows with missing values
    """

    # columns required for this ML task
    required_columns = ["Survived", "Pclass", "Sex", "Age", "Fare"]

    # ensure dataset contains expected columns
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_columns].copy()

    # convert categorical Sex → numeric
    sex_map = {"male": 0, "female": 1}
    df["Sex"] = df["Sex"].map(sex_map)

    # drop rows with missing values
    df = df.dropna()

    return df


def load_titanic():
    """
    Load and preprocess Titanic dataset, returns features X and target y.
    """

    df = load_raw_data()
    df = preprocess_titanic(df)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    return X, y