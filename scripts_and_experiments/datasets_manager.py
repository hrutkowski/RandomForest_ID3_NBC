import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def get_dataset_corona() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv('../datasets/corona.csv', low_memory=False)

    df = df.drop(['Ind_ID', 'Test_date'], axis=1)

    pd.set_option('display.max_columns', None)

    # Usuwanie NaN z kolumn
    for column in df.columns:
        df = df.dropna(subset=[column])

    for i in ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache']:
        df[i] = df[i].replace({True: 1, False: 0})

    df['Age_60_above'] = df['Age_60_above'].replace({'No': 0, 'Yes': 1})
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True, dtype=int)

    le = LabelEncoder()
    for i in ['Corona', 'Known_contact']:
        df[i] = le.fit_transform(df[i])

    df = df.astype('int32')

    """
    for column in df.columns:
        none_count = df[column].isna().sum()

        if none_count > 0:
            print(f"Column '{column}' has {none_count} occurrences of None.")
        else:
            print(f"Column '{column}' has no occurrences of None.")
    """

    X = df.drop(['Corona'], axis=1)
    y = df['Corona']

    return X, y


def get_dataset_divorce() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("../datasets/divorce.csv")

    X = df.drop(['Divorce_Y_N'], axis=1)
    y = df['Divorce_Y_N']

    return X, y


def get_dataset_glass() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("../datasets/glass.csv")

    num_bins = 6

    # Dyskretyzowanie ciągłych wartości w kolumnach na 6 przedziałow
    for col in df.columns[:-1]:
        df[col] = pd.qcut(df[col], q=num_bins, labels=False, precision=0, duplicates='drop')

    X = df.drop(['Type'], axis=1)
    y = df['Type']

    return X, y


def get_dataset_loan_approval() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("../datasets/loan_approval.csv")

    df = df.drop(['loan_id'], axis=1)

    label_encoder = LabelEncoder()
    for i in [' education', ' self_employed', ' loan_status']:
        df[i] = label_encoder.fit_transform(df[i])

    unique_counts = df.nunique()

    # Dyskretyzowanie ciągłych wartości w kolumnach

    columns_to_transform = [' income_annum', ' loan_amount', ' loan_term', ' cibil_score',
                            ' residential_assets_value', ' commercial_assets_value',
                            ' luxury_assets_value', ' bank_asset_value']

    for column in columns_to_transform:
        df[column] = pd.qcut(df[column], q=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=False)

    X = df.drop([' loan_status'], axis=1)
    y = df[' loan_status']

    return X, y


def get_class_distribution_for_dataset(get_dataset_function):
    X, y = get_dataset_function
    sns.countplot(x=y)
    plt.title('Class Distribution')
    plt.show()


#get_class_distribution_for_dataset(get_dataset_loan_approval())
