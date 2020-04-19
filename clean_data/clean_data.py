import pandas as pd
import click
from pathlib import Path

def _save_datasets(train, test, outdir: Path, flag):
    """save cleaned train and test datasets and write SUCCESS flag."""
    out_train = outdir / 'train_cleaned.csv/'
    out_test = outdir / 'test_cleaned.csv/'
    flag = outdir / flag

    train.to_csv(str(out_train), index=False)
    test.to_csv(str(out_test), index=False)

    flag.touch()


def CleanData(df, drop_columns, target_name):
    """clean data by dropping unnecessary columns, duplicate rows and empty rows of target.

        Parameters
        ----------
        df: dataframe
            dataframe object to be cleaned
        drop_columns: list of strings
            column names to be dropped
        target_name: string
            name of the target variable

        Returns
        -------
        cleaned dataframe
        """

    interim_df = df.drop(columns=drop_columns)

    interim_df_2 = (interim_df
                    .drop_duplicates(ignore_index=True))

    cleaned_df = (interim_df_2
                  .dropna(subset=[target_name],
                          how="any")
                  .reset_index(drop=True))

    return cleaned_df

@click.command()
@click.option('--in-train-csv')
@click.option('--in-test-csv')
@click.option('--out-dir')
@click.option('--flag')
def clean_datasets(in_train_csv, in_test_csv, out_dir, flag):
    # create directory
    out_dir = Path(out_dir)

    # load data
    train_df = pd.read_csv(in_train_csv)
    test_df = pd.read_csv(in_test_csv)

    # list of drop columns
    drop_columns = ["designation",
                    "winery",
                    "region_2",
                    "taster_twitter_handle"]

    # clean data
    cleaned_train = CleanData(train_df, drop_columns, "points")
    cleaned_test = CleanData(test_df, drop_columns, "points")

    _save_datasets(cleaned_train, cleaned_test, out_dir, flag)

if __name__ == '__main__':
    clean_datasets()