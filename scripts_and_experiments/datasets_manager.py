import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def get_dataset_corona() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv('../datasets/corona.csv', low_memory=False)

    df = df.drop(['Ind_ID', 'Test_date'], axis=1)
    #df.drop(df[df['Corona'] == 'other'].index, inplace=True)
    pd.set_option('display.max_columns', None)

    corona_column = df['Corona']

    class_counts = corona_column.value_counts()
    print(class_counts)

    # Usuwanie NaN z kolumn
    for column in df.columns:
        df = df.dropna(subset=[column])

    for i in ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache']:
        df[i] = df[i].replace({True: 1, False: 0})

    df['Age_60_above'] = df['Age_60_above'].replace({'No': 0, 'Yes': 1})
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True, dtype=int)

    le = LabelEncoder()
    for i in ['Corona', 'Known_contact']:
        print(f"Before encoding {i} values:")
        print(df[i].value_counts())  # Print the original values before encoding

        df[i] = le.fit_transform(df[i])  # Apply the LabelEncoder

        print(f"After encoding {i} values:")
        print(df[i].value_counts())  # Print the new values after encoding
        print("\n")

    df = df.astype('int32')

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


def get_dataset_letter_recognition() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("../datasets/letter-recognition.csv")
    #df = df.head(1000)

    le = LabelEncoder()
    for i in ['letter']:
        df[i] = le.fit_transform(df[i])

    X = df.drop(['letter'], axis=1)
    y = df['letter']

    return X, y


def get_class_distribution_for_dataset(get_dataset_function):
    X, y = get_dataset_function
    sns.countplot(x=y)
    plt.title('Class Distribution')
    plt.show()


def load_proper_dataset(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    if dataset_name == 'corona':
        X, y = get_dataset_corona()
    elif dataset_name == 'glass':
        X, y = get_dataset_glass()
    elif dataset_name == 'loan_approval':
        X, y = get_dataset_loan_approval()
    elif dataset_name == 'letter':
        X, y = get_dataset_letter_recognition()
    else:
        X, y = get_dataset_divorce()

    return X, y


def get_class_labels_for_dataset(dataset_name: str):
    if dataset_name == 'corona':
        df = pd.read_csv('../datasets/corona.csv', low_memory=False)
        unique_values = df['Corona'].unique()
    elif dataset_name == 'glass':
        df = pd.read_csv("../datasets/glass.csv")
        unique_values = df['Type'].unique()
    elif dataset_name == 'loan_approval':
        df = pd.read_csv("../datasets/loan_approval.csv")
        unique_values = df[' loan_status'].unique()
    elif dataset_name == 'letter':
        df = pd.read_csv("../datasets/letter-recognition.csv")
        unique_values = df['letter'].unique()
    else:
        df = pd.read_csv("../datasets/divorce.csv")
        unique_values = df['Divorce_Y_N'].unique()

    return unique_values
