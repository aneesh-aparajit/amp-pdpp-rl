import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from config import CFG


def get_dataset(exp_name: str) -> Tuple[np.array, np.array]:

    if not os.path.exists(f'../artifacts/scalers/{exp_name}'):
        os.makedirs(f'../artifacts/scalers/{exp_name}')

    df_train_clin = pd.read_csv(
        "../kaggle/input/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv")
    df_train_pept = pd.read_csv(
        "../kaggle/input/amp-parkinsons-disease-progression-prediction/train_peptides.csv")
    df_train_prot = pd.read_csv(
        "../kaggle/input/amp-parkinsons-disease-progression-prediction/train_proteins.csv")

    patients = {}
    for e in range(1, 5):
        for m in [0, 6, 12, 24]:
            df_train_clin[f'updrs_{e}_plus_{m}_months'] = 0

    for patient in df_train_clin.patient_id.unique():
        temp = df_train_clin[df_train_clin.patient_id == patient]
        month_list = []
        month_windows = [0, 6, 12, 24]
        for month in temp.visit_month.values:
            month_list.append([month, month + 6, month + 12, month + 24])
        for month in range(len(month_list)):
            for x in range(1, 5):
                arr = temp[temp.visit_month.isin(
                    month_list[month])][f'updrs_{x}'].fillna(0).to_list()
                if len(arr) == 4:
                    for e, i in enumerate(arr):
                        m = month_list[month][0]
                        temp.loc[temp.visit_month == m, [
                            f'updrs_{x}_plus_{month_windows[e]}_months']] = i
                else:
                    temp = temp[~temp.visit_month.isin(month_list[month])]
        patients[patient] = temp

    formatted_clin = pd.concat(
        patients.values(), ignore_index=True).set_index('visit_id').iloc[:, 7:]
    protfeatures = df_train_prot.pivot(
        index='visit_id', columns='UniProt', values='NPX')
    protfeatures.columns = [x+'_prot' for x in protfeatures.columns]
    peptfeatures = df_train_pept.pivot_table(
        index='visit_id', columns='UniProt', values='PeptideAbundance', aggfunc='mean')
    peptfeatures.columns = [x+'_pept' for x in peptfeatures.columns]

    df = protfeatures.merge(formatted_clin, left_index=True,
                            right_index=True, how='right')
    print(
        f'\nNA values: {df[protfeatures.columns].isna().sum().sum()/(len(df)*len(protfeatures.columns)):.2%}')
    df['visit_month'] = df.reset_index().visit_id.str.split(
        '_').apply(lambda x: int(x[1])).values

    df = peptfeatures.merge(df, left_index=True, right_index=True, how='right')
    print(
        f'\nNA values: {df[peptfeatures.columns].isna().sum().sum()/(len(df)*len(peptfeatures.columns)):.2%}')
    df['visit_month'] = df.reset_index().visit_id.str.split(
        '_').apply(lambda x: int(x[1])).values

    visit_month_list = df.reset_index().visit_id.str.split(
        '_').apply(lambda x: int(x[1])).unique().tolist()
    X = df.drop(formatted_clin.columns, axis=1)
    y = df[formatted_clin.columns]
    # print('\nX and y shapes: {}')

    def random_sample_imputation(df, var):
        random_sample = df[var].dropna().sample(
            df[var].isna().sum(), random_state=42, replace=True)
        random_sample.index = df[df[var].isna()].index
        df.loc[df[var].isna(), var] = random_sample
        return df

    for col in X.columns:
        X = random_sample_imputation(X, col)

    scaler_X = preprocessing.MinMaxScaler()
    scaler_Y = preprocessing.MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_Y.fit_transform(y)
    y_scaled = y_scaled[:, 0]

    # pickle.dump(scaler_X, open(f'../artifacts/scalers/{exp_name}/X.pkl', 'r'))
    # pickle.dump(scaler_Y, open(f'../artifacts/scalers/{exp_name}/y.pkl', 'r'))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, y_scaled, test_size=CFG.test_size)

    return X_train, X_test, y_train, y_test
