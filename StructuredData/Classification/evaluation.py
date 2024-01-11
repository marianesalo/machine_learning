import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    accuracy_score,
)
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve


def print_df_target_proportions(df: pd.DataFrame, target: str = "target"):
    """
    prints df shape and target proportions
    -----------------------------------------------------------------
    :param [pandas.DataFrame] df: dataframe object to be analysed
    :param [string] target: string with the name of the target column
    """

    print("shape: ", df.shape)
    for label in df[target].unique():
        print(
            "Proportion of " + str(label) + ": ",
            df[df[target] == label].shape[0] / df.shape[0],
            )


def calculate_performance(
        y_train: pd.Series,
        y_test: pd.Series,
        y_train_pred: pd.Series,
        y_test_pred: pd.Series,
        y_train_pred_proba: pd.Series,
        y_test_pred_proba: pd.Series,
):
    """
    Calculate classification metrics
    -----------------------------------------------------------------------------------------
    :param [pandas.Series] y_test: actual values of y from the test set
    :param [pandas.Series] y_test_pred_proba: predicted proba value for y from the test set
    :param [pandas.Series] y_test_pred: predicted value(1 or 0) for y from the test set
    :param [pandas.Series] y_train: actual values of y from the train set
    :param [pandas.Series] y_train_pred_proba: predicted proba value for y from the train set
    :param [pandas.Series] y_train_pred: predicted value(1 or 0) for y from the train set
    """

    print("--- classification report for train predictions ---")
    print(classification_report(y_train, y_train_pred))
    print("--- classification report for test predictions ---")
    print(classification_report(y_test, y_test_pred))
    print("AUPR train: ", average_precision_score(y_train, y_train_pred_proba))
    print("AUPR test: ", average_precision_score(y_test, y_test_pred_proba))
    print("---------------------------------------------------")
    print("Accuracy train: ", accuracy_score(y_train, y_train_pred))
    print("Accuracy test: ", accuracy_score(y_test, y_test_pred))
    print("AUC - roc train: ", roc_auc_score(y_train, y_train_pred_proba))
    print("AUC - roc test: ", roc_auc_score(y_test, y_test_pred_proba))


def plot_model_precision_recall_curve(
        y_test: pd.Series,
        y_test_pred_proba: pd.Series,
        y_train: pd.Series,
        y_train_pred_proba: pd.Series,
):
    """
    Plot precision-recall curve
    -----------------------------------------------------------------------------------------
    :param [pandas.Series] y_test: actual values of y from the test set
    :param [pandas.Series] y_test_pred_proba: predicted proba value for y from the test set
    :param [pandas.Series] y_train: actual values of y from the train set
    :param [pandas.Series] y_train_pred_proba: predicted proba value for y from the train set
    :return: matplotlib.pyplot.figure with the plot
    """

    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    average_precision_train = average_precision_score(y_train, y_train_pred_proba)
    precision_train, recall_train, thresholds_train = precision_recall_curve(
        y_train, y_train_pred_proba
    )
    axs[0].plot(recall_train, precision_train)
    axs[0].set_title(
        "Train dataset - Precision-Recall curve: AP={0:0.3f}".format(
            average_precision_train
        )
    )

    average_precision = average_precision_score(y_test, y_test_pred_proba)
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred_proba)
    axs[1].plot(recall, precision)
    axs[1].set_title(
        "Test dataset - Precision-Recall curve: AP={0:0.3f}".format(average_precision)
    )

    return f


def plot_model_roc_curve(y_test, y_test_pred_proba, y_train, y_train_pred_proba):
    """
    Plot roc curve
    -----------------------------------------------------------------------------------------
    :param [pandas.Series] y_test: actual values of y from the test set
    :param [pandas.Series] y_test_pred_proba: predicted proba value for y from the test set
    :param [pandas.Series] y_train: actual values of y from the train set
    :param [pandas.Series] y_train_pred_proba: predicted proba value for y from the train set
    :return: matplotlib.pyplot.figure with the plot
    """

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    # test plot
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)
    ax[1].plot(fpr, tpr)
    ax[1].set_title("Test dataset - ROC curve: " "AP={0:0.3f}".format(roc_auc_test))
    ax[1].set_ylabel("tpr")
    ax[1].set_xlabel("fpr")

    # train plot
    fpr, tpr, _ = roc_curve(y_train, y_train_pred_proba)
    roc_auc_train = roc_auc_score(y_train, y_train_pred_proba)
    ax[0].plot(fpr, tpr)
    ax[0].set_title("Train dataset - ROC curve: " "AP={0:0.3f}".format(roc_auc_train))
    ax[0].set_ylabel("tpr")
    ax[0].set_xlabel("fpr")

    return f


def prepare_df_for_deciles_analysis(y_test, y_test_pred_proba):
    """
    Prepare a dataset to analyse performance per deciles
    -----------------------------------------------------------------------------------------
    :param [pandas.Series] y_test: actual values of y from the test set
    :param [pandas.Series] y_test_pred_proba: predicted proba value for y from the test set
    :return: [pandas.DataFrame] analysis:
    """
    # compile predictions results on a dataset
    predictions = pd.DataFrame(
        {"decil": np.floor(y_test_pred_proba * 10), "y_actual": y_test}
    )

    precision = (
        predictions.groupby("decil")
        .sum()
        .reset_index()
        .sort_values("decil")
        .rename({"y_actual": "num_pos_actual"}, axis=1)
    )

    # compile volume of observations per decile
    public = (
        predictions.groupby("decil")
        .count()
        .reset_index()[["decil", "y_actual"]]
        .rename({"y_actual": "num_instances"}, axis=1)
    )
    public.loc[:, "perc_instances"] = public.num_instances / len(y_test)

    # analysis dataset with all previous information
    analysis = pd.merge(precision, public, on="decil", how="inner")
    analysis.loc[:, "num_neg_actual"] = analysis.num_instances - analysis.num_pos_actual
    analysis.loc[:, "adds_percentage"] = (
            analysis.num_pos_actual / analysis.num_instances
    )

    return analysis


def plot_percentiles_graph(df_decils):
    """
    plots a bar graph with the proportion of occurrences in each score bucket
    plots a line graph with the proportion of True Positive from the total of each bucket
    ----------------------------------------------------------------------------------------------
    :param [pandas.DataFrame] df_decils: df returned by prepare_df_for_deciles_analysis() function
    :return: matplotlib.pyplot.figure with the plot
    """
    f = plt.figure()
    width = 0.35
    barplot = df_decils["perc_instances"].plot(
        kind="bar", width=width, color="cornflowerblue", label="instances"
    )
    lineplot = df_decils["adds_percentage"].plot(
        secondary_y=True, color="green", label="added"
    )
    plt.title("Prediction Percentile Analysis: Added proportion vs Weight Proportion")
    plt.xlabel("Score predicted")
    barplot.set_ylabel("Proportion of observations predicted in this decil")
    lineplot.set_ylabel("Proportion of added skills within this decil")
    barplot.legend(loc=(1.04, -0.1))
    lineplot.legend(loc=(1.04, -0.2))

    create_bar_annotations(barplot)
    # Create line annotations
    for index, row in df_decils.iterrows():
        lineplot.text(
            index,
            row["adds_percentage"],
            "{:.2f}%".format(row["adds_percentage"] * 100),
            horizontalalignment="right",
        )

    return f


def create_bar_annotations(plot_object):
    for item in plot_object.patches:
        plot_object.annotate(
            format(item.get_height(), ".2f"),
            (item.get_x() + item.get_width() / 2, item.get_height()),
            ha="center",
            va="center",
            size=10,
            xytext=(0, 8),
            textcoords="offset points",
        )


def prepare_df_precision_recall_analysis(analysis):
    """
    Create a df to check the precision recall trade-of for each pedicting threshold
    --------------------------------------------------------------------------------------
    :param [pandas.DataFrame] analysis: df generated by prepare_df_for_deciles_analysis()
    :return: [pandas.DataFrame] acc_analysis
    """
    acc_analysis = pd.DataFrame()
    for decil in analysis.decil:
        row = (
            analysis[analysis.decil >= decil]
            .sum()
            .drop(["decil", "perc_instances", "adds_percentage"])
        )
        row["decil"] = decil
        acc_analysis = pd.concat([acc_analysis, pd.DataFrame([row])], ignore_index=True)

    total_observations = acc_analysis.num_instances[0]
    acc_analysis.loc[:, "perc_instances"] = (
            acc_analysis.num_instances / total_observations
    )
    acc_analysis.loc[:, "acc_precision"] = (
            acc_analysis.num_pos_actual / acc_analysis.num_instances
    )
    acc_analysis.loc[:, "acc_positive_coverage"] = acc_analysis.num_pos_actual / sum(
        analysis.num_pos_actual
    )
    acc_analysis.loc[:, "f1"] = (
            2
            * acc_analysis.acc_precision
            * acc_analysis.acc_positive_coverage
            / (acc_analysis.acc_precision + acc_analysis.acc_positive_coverage)
    )
    acc_analysis.head(10)

    return acc_analysis


def plot_acc_percentiles_graph(acc_analysis):
    """
    Considering success all the observations with score bigger os equal the score bucket:
        plots a bar graph with the proportion of occurrences in each score bucket (accumulated)
        plots a line graph with the proportion of True Positive from the total of each bucket (accumulated)
    -------------------------------------------------------------------------------------------------------
    :param [pandas.DataFrame] acc_analysis: dataframe returned from prepare_df_precision_recall_analysis()
    :return: matplotlib.pyplot.figure with the plot
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(acc_analysis.decil, acc_analysis.acc_precision, label="precision")
    axs[0].plot(acc_analysis.decil, acc_analysis.acc_positive_coverage, label="recall")
    axs[0].plot(acc_analysis.decil, acc_analysis.f1, label="f1")
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    axs[0].set_title("Accumulated Percentil Analysis")
    axs[0].set_xticks(acc_analysis["decil"])

    axs[1] = acc_analysis["perc_instances"].plot(
        kind="bar", color="lavender", label="instances", secondary_y=True
    )
    axs[1].set_title("Proportion of the test base that would fall into the percentile")
    axs[1].set_xticks(acc_analysis.index, acc_analysis["decil"])

    for index, row in acc_analysis.iterrows():
        axs[0].text(
            row["decil"],
            row["f1"],
            "{:.2f}".format(row["f1"]),
            horizontalalignment="center",
        )
        axs[0].text(
            row["decil"],
            row["acc_precision"],
            "{:.2f}".format(row["acc_precision"]),
            horizontalalignment="center",
        )
        axs[0].text(
            row["decil"],
            row["acc_positive_coverage"],
            "{:.2f}".format(row["acc_positive_coverage"]),
            horizontalalignment="center",
        )
        axs[1].text(
            index,
            row["perc_instances"],
            "{:.2f}".format(row["perc_instances"]),
            horizontalalignment="center",
        )

    return fig
