import pandas as pd
import click
from pathlib import Path
from sklearn.impute import SimpleImputer

def _save_datasets(train_features,
                   test_features,
                   train_target,
                   test_target,
                   outdir: Path,
                   flag):
    """save imputed and ML-ready train and target features
    # test features and target and write SUCCESS flag."""
    out_train_features = outdir / 'train_features.csv/'
    out_test_features = outdir / 'test_features.csv/'
    out_train_target = outdir / 'train_target.csv/'
    out_test_target = outdir / 'test_target.csv/'
    flag = outdir / flag

    train_features.to_csv(str(out_train_features),
                          index=False)
    test_features.to_csv(str(out_test_features),
                         index=False)
    train_target.to_csv(str(out_train_target),
                        index=False)
    test_target.to_csv(str(out_test_target),
                       index=False)

    flag.touch()

def ImputeWithConstant(train_df, test_df, cols=["taster_name"]):
    """function to impute taster_name
    # with 0 stands for "Unknown taster" """

    train_df = pd.DataFrame(train_df[cols])
    test_df = pd.DataFrame(test_df[cols])

    # create imputer object
    constant_imputer = SimpleImputer(strategy="constant",
                                     fill_value=0)

    # fit on the training set
    constant_imputer.fit(train_df)

    # transform training and test sets
    imputed_train_set = constant_imputer.transform(train_df)
    imputed_train_df = pd.DataFrame(imputed_train_set,
                                    columns=train_df.columns)

    imputed_test_set = constant_imputer.transform(test_df)
    imputed_test_df = pd.DataFrame(imputed_test_set,
                                   columns=test_df.columns)

    return imputed_train_df, imputed_test_df

def ImputeWithMedian(train_df, test_df, cols=["price", "year"]):
    """function to impute price and year
    # columns with the median value of each"""

    # separate to-be-imputed columns
    train_df = pd.DataFrame(train_df[cols])
    test_df = pd.DataFrame(test_df[cols])

    # create imputer object
    median_imputer = SimpleImputer(strategy="median")

    # fit on the training set
    median_imputer.fit(train_df)

    # transform training and test sets
    imputed_train_set = median_imputer.transform(train_df)
    imputed_train_df = pd.DataFrame(imputed_train_set,
                                    columns=train_df.columns)

    imputed_test_set = median_imputer.transform(test_df)
    imputed_test_df = pd.DataFrame(imputed_test_set,
                                   columns=test_df.columns)

    return imputed_train_df, imputed_test_df

def ImputeWithMostFrequent(train_df, test_df,
                           cols=["country", "province", "region_1", "variety"]):
    """function to impute country, province, region_1, variety
    # columns with the most_frequent value of each feature"""

    # separate to-be-imputed columns
    train_df = pd.DataFrame(train_df[cols])
    test_df = pd.DataFrame(test_df[cols])

    # create imputer object
    most_frequent_imputer = SimpleImputer(strategy="most_frequent")

    # fit on the training set
    most_frequent_imputer.fit(train_df)

    # transform training and test sets
    imputed_train_set = most_frequent_imputer.transform(train_df)
    imputed_train_df = pd.DataFrame(imputed_train_set,
                                    columns=train_df.columns)

    imputed_test_set = most_frequent_imputer.transform(test_df)
    imputed_test_df = pd.DataFrame(imputed_test_set,
                                   columns=test_df.columns)

    return imputed_train_df, imputed_test_df

def ImputeMissingValues(train_df, test_df):
    # separate non-NA cols
    is_features = [col for col in train_df.columns
                   if col.find("is_") != -1]
    interim_train_1 = train_df[is_features]
    interim_test_1 = test_df[is_features]

    # impute taster_name NA with 0 as "Unknown"
    constant_impute = ImputeWithConstant(train_df,
                                         test_df)
    interim_train_2 = constant_impute[0]
    interim_test_2 = constant_impute[1]

    # impute year and price with median
    median_impute = ImputeWithMedian(train_df,
                                     test_df)
    interim_train_3 = median_impute[0]
    interim_test_3 = median_impute[1]

    # impute country, province, region_1,
    # variety with most_frequent
    most_frequent_impute = ImputeWithMostFrequent(train_df,
                                                  test_df)
    interim_train_4 = most_frequent_impute[0]
    interim_test_4 = most_frequent_impute[1]

    # add partial features dataframes back
    # separate features from target
    train_features = (interim_train_4
                      .join(interim_train_3)
                      .join(interim_train_2)
                      .join(interim_train_1))

    test_features = (interim_test_4
                     .join(interim_test_3)
                     .join(interim_test_2)
                     .join(interim_test_1))

    # create train and test target dataframes
    train_target = pd.DataFrame(
        train_df["points"])

    test_target = pd.DataFrame(
        test_df["points"])

    return train_features, train_target, test_features, test_target

@click.command()
@click.option('--in-train-csv')
@click.option('--in-test-csv')
@click.option('--out-dir')
@click.option('--flag')
def impute_datasets(in_train_csv, in_test_csv, out_dir, flag):
    # create directory
    out_dir = Path(out_dir)

    # load data
    train_df = pd.read_csv(in_train_csv)
    test_df = pd.read_csv(in_test_csv)

    # perform imputation and separation of features and target
    imputed_data = ImputeMissingValues(train_df,
                                       test_df)
    train_features = imputed_data[0]
    train_target = imputed_data[1]
    test_features = imputed_data[2]
    test_target = imputed_data[3]

    _save_datasets(train_features,
                   test_features,
                   train_target,
                   test_target,
                   out_dir,
                   flag)

if __name__ == '__main__':
    impute_datasets()