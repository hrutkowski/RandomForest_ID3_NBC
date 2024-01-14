import pandas as pd
from typing import Tuple

from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder


#def get_dataset_corona() -> Tuple[pd.DataFrame, pd.Series]:

def get_dataset_corona() -> DataFrame:
    df = pd.read_csv('../datasets/corona.csv')

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

    # X = df.drop(['Corona'], axis=1)
    # y = df['Corona']

    return df


def get_dataset_divorce() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("datasets/divorce.csv")

    X = df.drop(['Divorce_Y_N'], axis=1)
    y = df['Divorce_Y_N']

    return X, y


def get_dataset_glass() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("datasets/glass.csv")

    X = df.drop(['Type'], axis=1)
    y = df['Type']

    return X, y


def get_dataset_loan_approval_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("datasets/loan_approval_dataset.csv")

    df = df.drop(['loan_id'], axis=1)

    X = df.drop(['loan_status'], axis=1)
    y = df['loan_status']

    return X, y
