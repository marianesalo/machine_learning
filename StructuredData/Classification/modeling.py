import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    make_scorer,
    auc
)
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from typing import List


def evaluate_model_type(X_train: pd.DataFrame, y_train: pd.Series, model_type_names: List[str], model_types_obj):
    """
    Perform k-fold cross validation with k=10 for each model in model_types_obj.
    ----------------------------------------------------------------------------------------------------------
    :param [pandas.DataFrame] X_train: df with independent variables values
    :param [pandas.Series] y_train: dependent variable values
    :param model_type_names: list of strings containing model names to be tested
    :param model_types_obj: list of model objects to be tested
    :return: df with average performance (f1 and average_precision) on the training and validation sets and
    the average time spent training and predicting.
    """

    avg_train = []
    avg_val = []
    avg_time = []
    avg_f1_train = []
    avg_f1_val = []
    i = 0
    for model in model_types_obj:
        print(model_type_names[i])
        skf = StratifiedKFold(n_splits=10, random_state=4, shuffle=True)
        skf.get_n_splits(X_train, y_train)
        performance_train = []
        performance_val = []
        f1s_train = []
        f1s_val = []
        performance_time = []
        for train_index, val_index in skf.split(X_train, y_train):
            X_train_cv, X_val = (
                X_train.to_numpy()[train_index],
                X_train.to_numpy()[val_index],
            )
            y_train_cv, y_val = (
                y_train.to_numpy()[train_index],
                y_train.to_numpy()[val_index],
            )
            start = time.time()
            model.fit(X_train_cv, y_train_cv)
            y_train_cv_proba = model.predict_proba(X_train_cv)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
            y_train_cv_pred = model.predict(X_train_cv)
            y_val_pred = model.predict(X_val)
            stop = time.time()
            time_spent = stop - start
            metric_train = average_precision_score(y_train_cv, y_train_cv_proba)
            metric_val = average_precision_score(y_val, y_val_proba)
            f1_train = f1_score(y_train_cv, y_train_cv_pred)
            f1_val = f1_score(y_val, y_val_pred)
            performance_train.append(metric_train)
            performance_val.append(metric_val)
            f1s_train.append(f1_train)
            f1s_val.append(f1_val)
            performance_time.append(time_spent)

        avg_performance_train = np.mean(performance_train)
        avg_performance_val = np.mean(performance_val)
        avg_f1score_train = np.mean(f1s_train)
        avg_f1score_val = np.mean(f1s_val)
        avg_performance_time = np.mean(performance_time)
        avg_train.append(avg_performance_train)
        avg_val.append(avg_performance_val)
        avg_f1_train.append(avg_f1score_train)
        avg_f1_val.append(avg_f1score_val)
        avg_time.append(avg_performance_time)
        i += 1

    df_models = pd.DataFrame(
        data={
            "model": model_type_names,
            "aucpr_train": avg_train,
            "aucpr_val": avg_val,
            "f1_train": avg_f1_train,
            "f1_val": avg_f1_val,
            "time": avg_time,
        }
    )
    return df_models


def chose_undersample_ratio(X: pd.DataFrame, y: pd.Series, ratios: List[float], model):
    """
    Perform k-fold cross validation with k=10 for each undersample ratio provided at ratios
    Returns a df with average performance on the training and validation sets.
    Metrics to quantify performance are f1, precision, recall and average_precision.
    ----------------------------------------------------------------------------------------------------------
    :param X: df with independent variables values
    :param y: dependent variable values
    :param ratios: list of ratios to be tested at the undersampling technique.
        A ratio value is the weight of the majority class. It decides how many samples from the majority class to keep.
        If the ratio is 1 we will have the same number of samples from the majority and the minority class.
        If the ratio is 2 we will have the double of samples of majority class when compared to the minority class.
        num_majority_keep = int(num_minority * ratio)

    :param model: model object to be used to perform the test
    :return: Returns a df with average performance on the training and validation sets.
    """

    avg_train = []
    avg_val = []
    avg_f1_train = []
    avg_f1_val = []
    avg_recall_train = []
    avg_recall_val = []
    avg_precision_train = []
    avg_precision_val = []
    i = 0

    # Combine X and y horizontally
    df = pd.DataFrame(
        np.column_stack((X, y)),
        columns=[f"col{i}" for i in range(X.shape[1])] + ["target"],
    )

    minority_samples = df[df.iloc[:, -1] == 1]
    majority_samples = df[df.iloc[:, -1] == 0]
    num_minority = minority_samples.shape[0]

    for ratio in ratios:
        print(f"----------- Calculating metrics for ratio: {ratio} -----------")
        # creates undersample
        num_majority_keep = int(num_minority * ratio)
        print("Num majority: ", num_majority_keep)
        print("Num minority: ", num_minority)
        undersampled_majority = resample(
            majority_samples, replace=False, n_samples=num_majority_keep, random_state=4
        )
        df_undersampled = (
            pd.concat([minority_samples, undersampled_majority], ignore_index=True)
            .sample(frac=1)
            .reset_index(drop=True)
        )
        print(df_undersampled.head(2))
        X_train = df_undersampled[:, :-1]
        y_train = df_undersampled[:, -1]
        print("Type of x: ", type(X_train))
        print("Type of y: ", type(y_train))
        # X_train = np.vstack([minority_samples, undersampled_majority])
        # y_train = np.concatenate([np.ones(minority_samples.shape[0]), np.zeros(undersampled_majority.shape[0])])

        # calculate performance for this undersample ratio
        skf = StratifiedKFold(n_splits=10, random_state=4, shuffle=True)
        skf.get_n_splits(X_train, y_train)
        performance_train = []
        performance_val = []
        f1s_train = []
        f1s_val = []
        precisions_train = []
        precisions_val = []
        recalls_train = []
        recalls_val = []
        n_cv = 1

        print("1 - Start Cross Validation")
        for train_index, val_index in skf.split(X_train, y_train):
            print("\t Number fold: ", n_cv)
            X_train_cv, X_val = X_train[train_index], X_train[val_index]
            y_train_cv, y_val = y_train[train_index], y_train[val_index]
            print("\t prepared x and y")
            print("\t Type of x: ", type(X_train_cv))
            print("\t Type of y: ", type(y_train_cv))
            model.fit(X_train_cv, y_train_cv)

            y_train_cv_proba = model.predict_proba(X_train_cv)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
            y_train_cv_pred = model.predict(X_train_cv)
            y_val_pred = model.predict(X_val)

            # calculate metrics
            metric_train = average_precision_score(y_train_cv, y_train_cv_proba)
            metric_val = average_precision_score(y_val, y_val_proba)
            f1_train = f1_score(y_train_cv, y_train_cv_pred)
            f1_val = f1_score(y_val, y_val_pred)
            precision_train = precision_score(y_train_cv, y_train_cv_pred)
            precision_val = precision_score(y_val, y_val_pred)
            recall_train = recall_score(y_train_cv, y_train_cv_pred)
            recall_val = recall_score(y_val, y_val_pred)

            # store metrics for fold
            performance_train.append(metric_train)
            performance_val.append(metric_val)
            f1s_train.append(f1_train)
            f1s_val.append(f1_val)
            precisions_train.append(precision_train)
            precisions_val.append(precision_val)
            recalls_train.append(recall_train)
            recalls_val.append(recall_val)
            n_cv += 1

        print("2 - Calculate mean performance for Cross validation")
        # calculate mean folds performance for the ratio
        avg_performance_train = np.mean(performance_train)
        avg_performance_val = np.mean(performance_val)
        avg_f1score_train = np.mean(f1s_train)
        avg_f1score_val = np.mean(f1s_val)
        avg_recall_train = np.mean(recalls_train)
        avg_recall_val = np.mean(recalls_val)
        avg_precision_train = np.mean(precisions_train)
        avg_precision_val = np.mean(precisions_val)

        # store ratio metrics
        avg_train.append(avg_performance_train)
        avg_val.append(avg_performance_val)
        avg_f1_train.append(avg_f1score_train)
        avg_f1_val.append(avg_f1score_val)
        avg_recall_train.append(avg_recall_train)
        avg_recall_val.append(avg_recall_val)
        avg_precision_train.append(avg_precision_train)
        avg_precision_val.append(avg_precision_val)
        i += 1

    df_ratios = pd.DataFrame(
        data={
            "ratio": ratios,
            "aucpr_train": avg_train,
            "aucpr_val": avg_val,
            "f1_train": avg_f1_train,
            "f1_val": avg_f1_val,
            "precision_train": avg_precision_train,
            "precision_val": avg_precision_val,
            "recall_train": avg_recall_train,
            "recall_val": avg_recall_val,
        }
    )
    return df_ratios


def plot_undersample_analysis(df_ratios: pd.DataFrame = None):
    """
    Plot performance for each undersample ratio
    ----------------------------------------------------------------------------------------------------------
    :param [pandas.DataFrame] df_ratios: df generated by chose_undersample_ratio
    :return: matplotlib.pyplot.figure with the plot
    """
    if not df_ratios:
        raise ValueError("No df_ratios were passed at the parameters")
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].plot(df_ratios.ratio, df_ratios.aucpr_train, label="auc_pr")
        axs[0].plot(df_ratios.ratio, df_ratios.recall_train, label="recall")
        axs[0].plot(df_ratios.ratio, df_ratios.precision_train, label="precision")
        axs[0].plot(df_ratios.ratio, df_ratios.f1_train, label="f1")
        handles, labels = axs[0].get_legend_handles_labels()
        # TODO lgd = CHECK LEGENDS
        axs[0].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1))
        axs[0].set_title("Ratios performance in train data")

        axs[1].plot(df_ratios.ratio, df_ratios.aucpr_val, label="auc_pr")
        axs[1].plot(df_ratios.ratio, df_ratios.recall_val, label="recall")
        axs[1].plot(df_ratios.ratio, df_ratios.precision_val, label="precision")
        axs[1].plot(df_ratios.ratio, df_ratios.f1_val, label="f1")
        axs[1].set_title("Ratios performance in validation data")

        plt.xticks(df_ratios["ratio"])
        for index, row in df_ratios.iterrows():
            # annotations for train plot
            axs[0].text(
                row["ratio"],
                row["f1_train"],
                "{:.2f}".format(row["f1_train"]),
                horizontalalignment="center",
            )
            axs[0].text(
                row["ratio"],
                row["precision_train"],
                "{:.2f}".format(row["precision_train"]),
                horizontalalignment="center",
            )
            axs[0].text(
                row["ratio"],
                row["recall_train"],
                "{:.2f}".format(row["recall_train"]),
                horizontalalignment="center",
            )
            axs[0].text(
                row["ratio"],
                row["aucpr_train"],
                "{:.2f}".format(row["aucpr_train"]),
                horizontalalignment="center",
            )
            # annotations for test plot
            axs[1].text(
                row["ratio"],
                row["f1_val"],
                "{:.2f}".format(row["f1_val"]),
                horizontalalignment="center",
            )
            axs[1].text(
                row["ratio"],
                row["precision_val"],
                "{:.2f}".format(row["precision_val"]),
                horizontalalignment="center",
            )
            axs[1].text(
                row["ratio"],
                row["recall_val"],
                "{:.2f}".format(row["recall_val"]),
                horizontalalignment="center",
            )
            axs[1].text(
                row["ratio"],
                row["aucpr_val"],
                "{:.2f}".format(row["aucpr_val"]),
                horizontalalignment="center",
            )

        return fig


def perform_hiperparameter_search(X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict = None):
    """
    Perform hiperparameter search
    ----------------------------------------------------------------------------------------------------------
    :param [pandas.DataFrame] X_train: df with independent variables values
    :param [pandas.Series] y_train: pd.Series with dependent variable values
    :param [dict] param_grid: parameter to be analysed
    :return: search object
    """

    # TODO: CHECK MAKE SCORER
    # precision_recall_auc = make_scorer(auc, greater_is_better=True, needs_proba=True, reorder=True)
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", verbosity=2, random_state=48
    )

    # define which parameters to test
    if not param_grid:
        param_grid = {
            "n_estimators": [3, 5, 7, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "max_depth": [4, 7, 9, 15],  # max depth of a tree
            "scale_pos_weight": [1, 5, 10, 20, 25, 50],
            "subsample": [0.5, 0.7, 1],  # Fraction of observations to be random sampled for each tree
            "colsample_bytree": [0.5, 0.7, 1],  # Fraction of features to be random sampled for each tree
            "min_child_weight": [1, 3, 5, 7],  # Minimum sum of weights of all observations required in a child
            "reg_alpha": [-1, 0, 1, 2, 5, 10],  # L1 regularization term
            "reg_lambda": [-1, 0, 1, 2, 5, 10],  # L2 regularization term
            "gamma": [-1, 0, 1, 2, 5, 10],  # Minimum loss reduction required to make a split
            "max_delta_step": [0, 1, 2, 5, 10],  # Maximum delta step each tree weight estimation can be
        }

    # instantiate search
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        scoring="f1",  # TODO make_sorer precision_recall_auc
        n_jobs=10,
        cv=10,
        return_train_score=True,
        random_state=42,
        verbose=2,
        n_iter=300,
    )

    # perform search
    search.fit(X_train, y_train)

    return search


def analyse_search_results(search):
    """
    Create a df with the Randomized search results and the difference between validation and train sets
    ----------------------------------------------------------------------------------------------------------
    :param search: RandomizedSearchCV or GridSearchCV object with param return_train_score=True
    :return df_search: pd.DataFrame with each search iteration results
    """

    df_search = pd.DataFrame(search.cv_results_)

    # Checking performance difference between validation and train sets
    df_search.loc[:, "diff_train_test"] = (
            df_search["mean_train_score"] - df_search["mean_test_score"]
    )
    df_search.loc[:, "diff_test_train"] = (
            df_search["mean_test_score"] - df_search["mean_train_score"]
    )
    df_search.loc[:, "round_test_score"] = df_search.round(2)["mean_test_score"].values

    # Order for best rounded test score and lowest difference between train and validation set
    cols_print = [
        "mean_train_score",
        "mean_test_score",
        "round_test_score",
        "rank_test_score",
        "diff_train_test",
    ]
    print("----- Top 10 performance on validation set -----")
    print(
        df_search.sort_values(
            ["round_test_score", "diff_test_train"], ascending=False
        ).head(10)[cols_print]
    )

    return df_search


def plot_search_distributions(df_search: pd.DataFrame, figsize: tuple = (20, 4)):
    """
    Plot distributions of the iterations concerning validation performance and train performance
    ----------------------------------------------------------------------------------------------------------
    :param [pandas.DataFrame] df_search: df generated by analyse_search_results(search)
    :param [tuple(int)] figsize: tuple with the figure size desired
    :return: matplotlib.pyplot.figure with the plot
    """
    # Checking distribution: validation score x difference between validation and train scores
    f, ax = plt.subplots(1, 3, figsize=figsize)

    ax[0].scatter(df_search.diff_train_test, df_search.mean_test_score)
    ax[0].axes.set_xlabel("difference")
    ax[0].axes.set_ylabel("test score")
    # ax[0].axes.set_xlim([-.3,0.1])
    ax[0].axes.set_ylim([0.4, 0.9])
    ax[0].set_title("Test score x Difference Train-Test")

    ax[1].scatter(df_search.mean_train_score, df_search.mean_test_score)
    ax[1].axes.set_xlabel("train score")
    ax[1].axes.set_ylabel("test score")
    # ax[1].axes.set_xlim([0,1.1])
    ax[1].axes.set_ylim([0.4, 0.9])
    ax[1].set_title("Test x Train score")

    temp = (
        df_search.groupby(["round_test_score"])
        .agg({"diff_train_test": np.min})
        .reset_index()
        .sort_values("round_test_score", ascending=False)
    )
    ax[2].scatter(temp.round_test_score, temp.diff_train_test)
    ax[2].axes.set_xlabel("test score")
    ax[2].axes.set_ylabel("diff")
    ax[2].axes.set_xlim([0.4, 0.9])
    # ax[2].axes.set_ylim([-0.2,0.05])
    ax[2].set_title("Difference Train-Test x Rounded test score")

    print(temp.head(10))

    return f
