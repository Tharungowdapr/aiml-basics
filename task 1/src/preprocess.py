

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Config ---
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "titanic.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
FIGURES_PATH = os.path.join(OUTPUT_PATH, "figures")
os.makedirs(FIGURES_PATH, exist_ok=True)

# --- Helper functions ---
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print("Loaded data shape:", df.shape)
    return df

def basic_info(df):
    print("\n--- Basic Info ---")
    print(df.info())
    print("\n--- Null counts ---")
    print(df.isnull().sum())
    print("\n--- Dtypes ---")
    print(df.dtypes)
    print("\n--- Head ---")
    print(df.head())

def missing_value_summary(df):
    pct = df.isnull().mean() * 100
    print("\n--- Missing value percentage ---")
    print(pct[pct > 0].sort_values(ascending=False))

def impute_missing(df):
    df = df.copy()
    # Example strategy for Titanic:
    # - Age: impute median
    # - Embarked: impute mode
    # - Fare: impute median if needed
    # - Cabin: drop or create flag (we'll create a flag + drop full Cabin text)
    df['Cabin_missing'] = df['Cabin'].isnull().astype(int)
    df = df.drop(columns=['Cabin'])

    # Numeric columns
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    # Exclude target and passenger id if desired
    if 'PassengerId' in num_cols:
        num_cols.remove('PassengerId')
    if 'Survived' in num_cols:
        num_cols.remove('Survived')

    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df

def encode_features(df):
    df = df.copy()
    # Example: One-hot encode 'Sex' and 'Embarked', label encode 'Title' if created
    # Create simple features
    df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    # Reduce rare titles
    rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')

    # Drop Name and Ticket (textual messy features)
    df = df.drop(columns=['Name', 'Ticket'])

    # Columns to one-hot encode
    ohe_cols = ['Sex', 'Embarked', 'Title']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
    return df

def scale_features(df, scaler='standard'):
    df = df.copy()
    # Choose numeric columns to scale (exclude target and PassengerId)
    target = 'Survived' if 'Survived' in df.columns else None
    if target:
        cols = df.drop(columns=[target, 'PassengerId']).select_dtypes(include=['int64','float64']).columns
    else:
        cols = df.drop(columns=['PassengerId']).select_dtypes(include=['int64','float64']).columns

    if scaler == 'standard':
        s = StandardScaler()
    else:
        s = MinMaxScaler()

    df[cols] = s.fit_transform(df[cols])
    return df

def detect_and_remove_outliers(df, numeric_cols=None, method='iqr', iqr_multiplier=1.5):
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    # Don't include target or PassengerId
    numeric_cols = [c for c in numeric_cols if c not in ('PassengerId','Survived')]

    if method == 'iqr':
        initial_shape = df.shape
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - iqr_multiplier * IQR
            upper = Q3 + iqr_multiplier * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        print(f"Removed outliers using IQR. {initial_shape} -> {df.shape}")
    else:
        print("Other outlier methods not implemented in this script.")
    return df

def visualize_outliers(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cols = [c for c in cols if c not in ('PassengerId','Survived')]
    for col in cols:
        plt.figure(figsize=(6,3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot: {col}")
        out_file = os.path.join(FIGURES_PATH, f"box_{col}.png")
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
    print(f"Saved boxplots to {FIGURES_PATH}")

def main():
    df = load_data()
    basic_info(df)
    missing_value_summary(df)

    df = impute_missing(df)
    print("\nAfter imputation, nulls:")
    print(df.isnull().sum().sum())

    df = encode_features(df)
    print("\nAfter encoding, columns:", df.columns.tolist())

    # Visualize outliers (before scaling)
    visualize_outliers(df)

    # Remove outliers
    df = detect_and_remove_outliers(df, method='iqr', iqr_multiplier=1.5)

    # Scale features
    df = scale_features(df, scaler='standard')

    # Save cleaned data
    out_file = os.path.join(OUTPUT_PATH, "clean_data.csv")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    df.to_csv(out_file, index=False)
    print("Cleaned data saved to:", out_file)

if __name__ == "__main__":
    main()
