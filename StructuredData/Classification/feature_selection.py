import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import shap


def check_null_values(df: pd.DataFrame, print_sum_df_nulls: bool = True):
    """
    Create a dataframe with the name of the column, the count of null values for the column and its proportion
    ___________________________________________________________________________________________________________
    :param [pandas.DataFrame] df: dataframe to be analysed
    :param [boolean] print_sum_df_nulls: if you want to print the df
    :return: df_nulls: df with the nulls statistics
    """

    df_nulls = df.isnull().sum().reset_index()
    df_nulls = df_nulls.rename({"index": "column", 0: "total_nulls"}, axis=1)
    df_nulls.loc[:, "proportion_nulls"] = df_nulls.total_nulls / df.shape[0]

    if print_sum_df_nulls:
        print("Total of nulls per column", df_nulls)

    return df_nulls


def create_correlation_matrix(df: pd.DataFrame, figsize: tuple = (20, 20)):
    """
    Create a correlation matrix between all columns of the dataframe
    ______________________________________________________________________________
    :param [pandas.DataFrame] df: dataframe to be analysed
    :param [tuple(int)] figsize: size of the figure
    :return: matplotlib.pyplot.figure with the plot
    """
    print("started correlation matrix")
    # calculate correlations
    df_corr = df.corr()

    # create a zeros matrix with the size of our correlation matrix
    mask = np.zeros_like(df_corr, dtype=bool)
    # make the upper triangle of the matrix = 1 (True)
    mask[np.triu_indices_from(mask)] = True

    # plot correlation heatmap
    f, ax = plt.subplots(figsize=figsize)
    heatpmat = sns.heatmap(
        df_corr,
        mask=mask,
        square=True,
        linewidths=0.6,
        cmap="coolwarm",
        cbar_kws={"shrink": 0.5, "ticks": [-1, -0.5, 0, 0.5, 1]},
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
    )
    # ax.set_xtickslabels(df_corr.columns.tolist())
    sns.set_style({"xtick.bottom": True}, {"ytick.left": True})

    print("finished correlation matrix")

    return f


def perform_boruta(X_train: pd.DataFrame, y_train: pd.Series, model=None):
    """
    Perform boruta and put the results in a dataframe
    ______________________________________________________________________________
    :param [pandas.DataFrame] X_train: dataframe with the features to be analysed
    :param [pandas.Series] y_train: target vector
    :param model: model object to be considered, if None use RandomForestClassifier
    :return: df_boruta: df with the boruta algorithm results
    """
    print("started boruta")
    if not model:
        model = RandomForestClassifier(random_state=48, n_estimators=10, max_depth=13)

    boruta = BorutaPy(model, n_estimators="auto", verbose=2, random_state=48)
    boruta.fit(np.array(X_train.fillna(0)), y_train.ravel())
    df_boruta = pd.DataFrame(
        {
            "variable": X_train.columns,
            "sup": boruta.support_,
            "sup_weak": boruta.support_weak_,
            "var_rank": boruta.ranking_,
        }
    )

    print(df_boruta.info())
    return df_boruta


def plot_feature_importances(model):
    """
    Plot feature importances for the model
    ______________________________________________________________________________
    :param model: model object to be considered
    :param [pandas.DataFrame] X_train: dataframe with the features to be analysed
    :param [int] n_show: number of features to show
    :return: matplotlib.pyplot.figure with the plot
    """
    from xgboost import plot_importance

    f = plot_importance(model)

    return f


def perform_shapely_importance(model, X_test: pd.DataFrame):
    """
    Plot shapely values
    ______________________________________________________________________________
    :param model: model object
    :param [pandas.DataFrame] X_test: df with the independent variables values
    :return: matplotlib.pyplot.figure with the plot
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    figure = shap.summary_plot(shap_values, plot_type="dot", show=False)

    return figure