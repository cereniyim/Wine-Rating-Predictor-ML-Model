import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error

import click
from pathlib import Path

import pickle

import logging

logging.basicConfig(level=logging.INFO)

def convert_features_to_array(features):
    """convert dataframe object to array.

        Parameters
        ----------
        features: dataframe
            features dataframe to be converted

        Returns
        -------
        array
        """
    num_rows = len(features)
    num_cols = len(features.columns)

    features_array = (np
                      .array(features)
                      .reshape((num_rows,
                                num_cols)))

    return features_array

def convert_target_to_array(target):
    """function to convert single-column dataframe
    # to a 1-D array"""

    target_array = np.array(target).reshape((-1,))
    return target_array

def PlotPredictedVSActual(predictions, actuals, png_name):
    """plot two subplots of two different arrays.

            Parameters
            ----------
            predictions: array
                prediction 1D array generated by ML model
            actual: array
                target values 1D array
            png_name: name of the png file to be saved

            Returns
            -------
            None
            """

    #set figure size and the font size
    plt.figure(figsize=(26, 12))
    plt.rcParams['font.size'] = 24

    # set histogram of predictions, title and labels
    ax = plt.subplot(121)
    ax.hist(predictions,
            bins=10,
            color = "#971539",
            edgecolor = 'white')
    ax.set_xlabel("points", size=24)
    ax.set_xticks(range(80, 101))
    ax.set_ylabel("count", size=24)
    ax.set_title("Predicted Distribution", size=24)
    plt.grid(b=True, axis='y', alpha=0.3)

    # histogram of actual values, title and labels
    ax2 = plt.subplot(122)
    ax2.hist(actuals,
             bins=20,
             color = "#971539",
             edgecolor = 'white')
    ax2.set_xlabel("points", size=24)
    ax2.set_xticks(range(80, 101))
    ax2.set_ylabel("count", size=24)
    ax2.set_title("Actual Distribution", size=24)
    plt.grid(b=True, axis='y', alpha=0.3)

    plt.savefig(png_name)

def PlotFeatureImportances(model, feature_names, png_name):
    """function to plot feature importances
    for the given model, feature_names as a list
    saves the plot to the given png_name
    create feature importances dataframe"""

    feature_importances = (pd
                           .DataFrame(
                               {'feature': feature_names,
                                'importance': model
                                .feature_importances_}))

    feature_importances = (feature_importances
                           .sort_values(by="importance",
                                        ascending=False))
    # set plot and font size
    plt.figure(figsize=(20, 10))
    plt.rcParams['font.size'] = 24
    sns.set(font_scale=1.5, style="whitegrid")

    # set color of the bars
    values = np.array(feature_importances.importance)
    colors = ["#808080" if (y < max(values))
              else "#971539" for y in values]

    # set the plot
    ax = sns.barplot(x="importance",
                     y="feature",
                     data=feature_importances,
                     palette=colors)

    # set title and save plot
    plt.title("Feature Importances", size =24)
    plt.savefig(png_name)


@click.command()
@click.option('--in-test-features-csv')
@click.option('--in-test-target-csv')
@click.option('--in-trained-model')
@click.option('--out-dir')
@click.option('--flag')
def evaluate_model(in_test_features_csv,
                   in_test_target_csv,
                   in_trained_model,
                   out_dir,
                   flag):

    log = logging.getLogger('evaluate-model')

    # create directory
    out_dir = Path(out_dir)

    # load model
    model = (pickle.load(open(in_trained_model, 'rb')))

    # load data and save feature names
    test_features = pd.read_csv(in_test_features_csv)
    test_target = pd.read_csv(in_test_target_csv)
    FEATURE_NAMES = test_features.columns

    # convert dataframes to arrays
    X_test = convert_features_to_array(test_features)
    y_test = convert_target_to_array(test_target)

    # generate predictions print error metric
    y_predicted = model.predict(X_test)
    mean_square_error = mean_squared_error(y_test, y_predicted)
    log.info("Mean square error of the model is: {}"
             .format(round(mean_square_error, 2)))

    # save plot of actual VS predicted distributions
    pred_actuals_png = out_dir / 'PredictionsVSActuals.png'
    PlotPredictedVSActual(y_predicted, y_test, pred_actuals_png)

    # save plot of feature importances
    feature_importance_png = out_dir / 'FeatureImportances.png'
    PlotFeatureImportances(model, FEATURE_NAMES, feature_importance_png)

    # create success flag
    flag = out_dir / flag
    flag.touch()

if __name__ == '__main__':
    evaluate_model()


