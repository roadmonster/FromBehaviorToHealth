import pandas as pd
from model_utils import *


if __name__ == "__main__":
    # Target (y) column
    y_var = 'DIABETE3'

    # Unused feature columns
    x_var_to_drop = [
        'DIABAGE2', 'PDIABTST', 'PREDIAB1', 'INSULIN',
        'BLDSUGAR', 'DOCTDIAB', 'CHKHEMO3', 'FEETCHK',
        'EYEEXAM1', 'DIABEYE', 'DIABEDU'
    ]

    # Load Data
    brfss_df = pd.read_csv('Data/LLCP2018_cleaned.csv.gz').sample(3000)
    var_type_df = pd.read_csv('Data/var_type.csv')

    # Preprocessing
    X_train, X_test, y_train, y_test = preprocess(brfss_df, var_type_df, y_var, x_var_to_drop)

    # Train
    clf = train(
        X_train, y_train,
        model_save_path='model_output/{}.joblib'.format(y_var),
        parameters={
            'max_depth': np.arange(4, 7),
            'n_estimators': [150, 250, 350],
            'learning_rate': [0.05, 0.1, 0.25],
            'colsample': [0.6, 0.7, 0.8]
        }
    )

    # Report
    print('Train:')
    report(clf, X_train, y_train)
    print('Test:')
    report(clf, X_test, y_test)

    # Interpret
    interpret_model(clf, X_train, output_folder='./model_output/{}/'.format(y_var))
