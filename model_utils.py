import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from joblib import dump, load
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report


def get_var_list_of_type(var_type_df, target_type):
    """Return variable list of given variable type in given variable type table."""
    return (
        var_type_df
        .loc[var_type_df['Type'] == target_type, 'Variable']
        .tolist()
    )


def preprocess(df, var_type_df, y_var, x_var_to_drop, test_size=0.25):
    """
    Preprocess data with following steps:
    1. Drop rows contains NA values in y column.
    2. Split X and y.
    3. Onehot encoding for multi-categorical variables.
    4. Split test and train.
    Returns X and y of train and test sets.
    """
    print('Preprocessing...')

    # Drop rows contains NA on y value
    df = df.dropna(subset=[y_var])

    # Drop y var and unused feature variables from var type table
    var_type_df = var_type_df.loc[
        (var_type_df['Variable'] != y_var) & (~var_type_df['Variable'].isin(x_var_to_drop))
    ]

    # Get y, target variable
    y = df[y_var]

    # Get one-hot encoding of multicategorical variables
    multicat_df = pd.get_dummies(
        df[get_var_list_of_type(var_type_df, 'Multicategorical')]
        .astype(str)
        .replace('nan', np.nan)
    )

    # Concat onehot encoding of multicategory with continuous and binary variables
    other_df = df[
        get_var_list_of_type(var_type_df, 'Continuous') +
        get_var_list_of_type(var_type_df, 'Binary')
    ].copy()
    X = pd.concat([multicat_df, other_df], axis=1)

    print('Number of records: {}\nNumber of features: {}'.format(*X.shape))

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print('Train size: {}\nTest size: {}\n'.format(len(y_train), len(y_test)))

    return X_train, X_test, y_train, y_test


def save_model(model, save_path):
    """Save model to given path."""
    # Create target folder if not exist
    if '/' in save_path:
        folder = save_path[:save_path.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Save model
    dump(model, save_path)


def xgb_grid_search(X_train, y_train, cv=5, scoring='f1', parameters=None):
    """
    Perform grid search for XGBoost classification model.
    Returns the best classifier.
    """
    # Create XGBoost Classifier
    xgb_model = XGBClassifier(objective='binary:logistic', verbosity=0)

    # Parameters to perform grid search
    if not parameters:
        parameters = {
            'max_depth': np.arange(6, 15),
            'n_estimators': np.arange(30, 220, 30),
            'learning_rate': np.logspace(-2, -0.05, 5),
            'colsample': np.arange(0.6, 1.0, 0.1)
        }

    # Create Grid search object with cross validation
    clf = GridSearchCV(
        xgb_model, parameters, n_jobs=-1, cv=cv,
        scoring=scoring, verbose=1, refit=True
    )

    # Perform grid search on training data
    print('Start Grid Search...')
    clf.fit(X_train, y_train)

    # Print best parameters and score
    print("Best parameter:")
    print(clf.best_params_)
    print("Best score: {:.4f}\n".format(clf.best_score_))

    return clf.best_estimator_


def train(
    X_train, y_train,
    model_save_path=None, retrain=False,
    cv=5, scoring='f1', parameters=None
):
    """Train or load existing model."""
    # Retrain flag on, always retrain model
    if retrain:
        clf = xgb_grid_search(
            X_train, y_train,
            cv=cv, scoring=scoring, parameters=parameters
        )

    # Try to load model from path, retrain on fail to load
    else:
        try:
            clf = load(model_save_path)
            clf.predict(X_train)         # Test whether the model works
            print('Existed model loaded from {}'.format(model_save_path))
        except Exception as _:
            clf = xgb_grid_search(
                X_train, y_train,
                cv=cv, scoring=scoring, parameters=parameters
            )

    # Save model
    if model_save_path:
        save_model(clf, model_save_path)
    return clf


def report(model, X, y_true):
    """Report model on accuracy, F1 score, and precision/recall table"""
    y_pred = model.predict(X)
    print('Accuracy: {}'.format(accuracy_score(y_true, y_pred)))
    print('F1 Score: {}'.format(f1_score(y_true, y_pred)))
    print(classification_report(y_true, y_pred))


def interpret_model(model, X, print_top=True, output_folder='.'):
    """
    Interpret the model by generating plots and tables on multiple feature importance scores.

    Scores including:
    1. mean(|Tree SHAP|)
       Each Tree SHAP value measures the change in the margin output of one feature on one record.
       mean(|SHAP|) of all SHAP values of one feature measures the overall impact of that feature.
       https://github.com/slundberg/shap
       http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
       https://arxiv.org/abs/1905.04610
    2. Gain
       The average training loss reduction gained when using a feature for splitting.
    3. Cover
       The number of times a feature is used to split the data across all trees
       weighted by the number of training data points that go through those splits.

    Also produces a SHAP distribution plot of top 20 variables,
    which helps to explain the correlation of the impact of variable vs. the value of variable.
    """
    # Create target folder if not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Mean absolute SHAP scores
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    feature_importance_df = pd.DataFrame(
        zip(X.columns, np.abs(shap_values).mean(axis=0)),
        columns=['variable', 'mean(|SHAP|)']
    )

    # Sort on mean abs SHAP score
    feature_importance_df = feature_importance_df.sort_values('mean(|SHAP|)', ascending=False)

    # Save mean abs SHAP score figure
    plt.figure(figsize=[8, 10])
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('Feature Importance, type="mean(|SHAP|)"')
    plt.savefig(
        os.path.join(output_folder, 'feature_importance_SHAP_mean_abs.png'),
        bbox_inches='tight'
    )

    # Save SHAP score distribution figure
    plt.figure(figsize=[8, 10])
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(
        os.path.join(output_folder, 'feature_importance_SHAP_distribution.png'),
        bbox_inches='tight'
    )

    # Feature importance from XGBoost - gain and cover
    for c in ['gain', 'cover']:
        # Get feature importance scores from model
        score_df = pd.DataFrame(
            model.get_booster().get_score(importance_type=c).items(),
            columns=['variable', c]
        )
        # Merge into feature importance score table
        feature_importance_df = (
            feature_importance_df
            .merge(score_df, how='left', on='variable')
            .fillna(0)
        )
        # Save plot
        fig, ax = plt.subplots(figsize=(8, 10))
        plot_importance(model, max_num_features=20, importance_type=c, ax=ax)
        ax.set_title('Feature Importance, type="{}"'.format(c))
        ax.set_xlabel('')
        fig.savefig(
            os.path.join(output_folder, 'feature_importance_{}.png'.format(c)),
            bbox_inches='tight'
        )

    # Save feature importance table to file
    feature_importance_df.to_csv(
        os.path.join(output_folder, 'feature_importance.csv'),
        index=False
    )

    # Print top 20 features with highest mean abs SHAP score
    if print_top:
        print(feature_importance_df.head(20))

    return feature_importance_df
