import pandas as pd
import numpy as np
import click
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import pickle

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
    to a 1-D array"""
    target_array = np.array(target).reshape((-1,))
    return target_array

@click.command()
@click.option('--in-train-features-csv')
@click.option('--in-train-target-csv')
@click.option('--out-dir')
def train_model(in_train_features_csv,
                in_train_target_csv,
                out_dir):
    # create directory
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    train_features = pd.read_csv(in_train_features_csv)
    train_target = pd.read_csv(in_train_target_csv)

    # convert dataframes to arrays
    X = convert_features_to_array(train_features)
    y = convert_target_to_array(train_target)

    # create random forest regressor model
    # with fine-tuned set of parameters
    model = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                                  max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                                  max_samples=None, min_impurity_decrease=0.0,
                                  min_impurity_split=None, min_samples_leaf=2,
                                  min_samples_split=4, min_weight_fraction_leaf=0.0,
                                  n_estimators=200, n_jobs=None, oob_score=False,
                                  random_state=42, verbose=0, warm_start=False)
    # train model
    model.fit(X, y)

    # save model to directory
    pickle.dump(model, open(out_dir / 'model.sav', 'wb'))

if __name__ == '__main__':
    train_model()