import pandas as pd
import click
from pathlib import Path
from category_encoders.ordinal import OrdinalEncoder

def _save_datasets(train, test, outdir: Path, flag):
    """save transformed train and test datasets and write SUCCESS flag."""
    # csv paths and flag path
    out_train = outdir / 'train_transformed.csv/'
    out_test = outdir / 'test_transformed.csv/'
    flag = outdir / flag

    # save as csv and create flag file
    train.to_csv(str(out_train), index=False)
    test.to_csv(str(out_test), index=False)

    flag.touch()

def EncodeCategoricalData(train_df, test_df):
    """encode data with OrdinalEncoder

            Parameters
            ----------
            train_df: dataframe
                training dataframe object to fit and transform
            test_df: dataframe
                test dataframe object to transform

            Returns
            -------
            transformed training and test dataframe
            """

    # column list to ordinal encode
    ordinal_encode_cols = ["country", "province",
                           "region_1", "taster_name", "variety"]

    # create ordinal encode object
    # object assigns -1 to the first-time-seen values of the test set
    ordinal_encoder = OrdinalEncoder(cols=ordinal_encode_cols,
                                     return_df=True,
                                     handle_unknown="value",
                                     handle_missing="return_nan")

    # fit object on the train dataset
    ordinal_encoder.fit(train_df)

    # transform train and test datasets
    ord_encoded_train = (ordinal_encoder
                         .transform(train_df))

    ord_encoded_test = (ordinal_encoder
                        .transform(test_df))

    return ord_encoded_train, ord_encoded_test

@click.command()
@click.option('--in-train-csv')
@click.option('--in-test-csv')
@click.option('--out-dir')
@click.option('--flag')
def transform_datasets(in_train_csv, in_test_csv, out_dir, flag):
    #crete directory
    out_dir = Path(out_dir)

    # load data
    train_df = pd.read_csv(in_train_csv)
    test_df = pd.read_csv(in_test_csv)

    # transform data
    encoded_data = EncodeCategoricalData(train_df, test_df)

    transformed_train = encoded_data[0]
    transformed_test = encoded_data[1]

    _save_datasets(transformed_train, transformed_test, out_dir, flag)

if __name__ == '__main__':
    transform_datasets()