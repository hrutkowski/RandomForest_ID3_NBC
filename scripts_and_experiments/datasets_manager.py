import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder


def get_dataset_corona() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv('../datasets/corona.csv', low_memory=False)

    df = df.drop(['Ind_ID', 'Test_date'], axis=1)

    pd.set_option('display.max_columns', None)

    # deleting all nan in df columns
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

    for column in df.columns:
        none_count = df[column].isna().sum()

        if none_count > 0:
            print(f"Column '{column}' has {none_count} occurrences of None.")
        else:
            print(f"Column '{column}' has no occurrences of None.")

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

    # columns_list = df.columns
    # for column in df.columns:
    #     unique_values = df[column].unique()
    #     #print(f"Unique values in column '{column}': {unique_values}")

    X = df.drop(['Divorce_Y_N'], axis=1)
    y = df['Divorce_Y_N']

    return X, y


def get_dataset_glass() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("datasets/glass.csv")

    X = df.drop(['Type'], axis=1)
    y = df['Type']

    return X, y


def get_dataset_loan_approval() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("datasets/loan_approval.csv")

    df = df.drop(['loan_id'], axis=1)

    X = df.drop(['loan_status'], axis=1)
    y = df['loan_status']

    return X, y
