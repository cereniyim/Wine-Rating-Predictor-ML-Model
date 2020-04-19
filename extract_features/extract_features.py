import pandas as pd
import click
from pathlib import Path
import re
import datetime

def _save_datasets(train, test, outdir: Path, flag):
    """save features extracted train and test datasets and write SUCCESS flag."""
    out_train = outdir / 'train_features_extracted.csv/'
    out_test = outdir / 'test_features_extracted.csv/'
    flag = outdir / flag

    train.to_csv(str(out_train), index=False)
    test.to_csv(str(out_test), index=False)

    flag.touch()

# description
def extract_features_from_description(df,
                                      new_feature_name,
                                      extract_words):
    """perform regex search in the description column.
    add new binary column if any of the extract words are found

         Parameters
         ----------
         df: dataframe
             dataframe object to be modified
         new_feature_name: string
             feature name to be added to the df
         extract_words: list of strings
             word list to be searched
         ASSUMPTION:There is no NA values exists

         Returns
         -------
         modified dataframe
         """

    check_regex = (r'\b(?:{})\b'
                   .format('|'
                           .join(
                               map(re.escape,
                                   extract_words))))

    df[new_feature_name] = (df['description']
                            .str
                            .contains(check_regex,
                                      regex=True)
                            .astype('uint8'))
    return df

# title
def extract_year_from_title(title_num_list):
    """assign first number found
    if the number is between 1900 and 2020
    else assign 0

             Parameters
             ----------
             title_num_list: list of integers
                 series object that contains list of integers
             ASSUMPTION:There is no NA values exists

             Returns
             -------
             integer (as year or 0)
             """

    int_list = []
    now = datetime.datetime.now()

    for item in title_num_list:
        int_list.append(int(item))

    for item in int_list:
        if item <= now.year and item >= 1900:
            return item
        else:
            return 0

# variety
def extract_blend_from_variety(variety):
    """search for multiple occurrences of variety or word "Blend".
    return 1 if found return 0 if not. Exception is Xarel-lo

             Parameters
             ----------
             variety: string
                 series object that contains string
             ASSUMPTION:There is no NA values exists

             Returns
             -------
             integer (1 or 0)
             """

    if (variety.find("-") != -1) | (variety.find("Blend") != -1):
        if variety == "Xarel-lo":
            return 0
        else:
            return 1
    else:
        return 0

@click.command()
@click.option('--in-train-csv')
@click.option('--in-test-csv')
@click.option('--out-dir')
@click.option('--flag')
def extract_features(in_train_csv, in_test_csv, out_dir, flag):
    # create directory
    out_dir = Path(out_dir)

    # load data
    train_df = pd.read_csv(in_train_csv)
    test_df = pd.read_csv(in_test_csv)

    # make a list values of
    # is_red, is_white, is_rose,
    # is_sparkling, is_dry, is_sweet
    is_red_list = ["red", "Red", "RED",
                   "noir", "NOIR", "Noir",
                   "black", "BLACK", "Black"]

    is_white_list = ["white", "WHITE", "White",
                     "blanc", "Blanc", "BLANC",
                     "bianco", "Bianco", "BIANCO",
                     "blanco", "Blanco", "BLANCO",
                     "blanca", "Blanca", "BLANCA"]

    is_rose_list = ["rose", "ROSE", "Rose",
                    "rosé", "Rosé", "ROSÉ"]

    is_sparkling_list = ["sparkling", "SPARKLING", "Sparkling"]

    is_dry_list = ["dry", "Dry", "DRY",
                   "dried", "Dried", "DRIED"]

    is_sweet_list = ["sweet", "Sweet", "SWEET"]

    desc_extracting_dict = {
        "is_red": is_red_list,
        "is_white": is_white_list,
        "is_rose": is_rose_list,
        "is_sparkling": is_sparkling_list,
        "is_dry": is_dry_list,
        "is_sweet": is_sweet_list
    }

    # add is_red, is_white, is_rose, is_sparkling, is_dry, is_sweet
    # to train and test datasets
    for key, value in desc_extracting_dict.items():
        interim_train = extract_features_from_description(
            train_df, key, value)
        interim_test = extract_features_from_description(
            test_df, key, value)


    # add year to train dataset
    interim_train["title_numlist"] = (interim_train
                                      .title
                                      .str
                                      .findall(r'\b\d+\b'))

    interim_train["year"] = (interim_train
                             .title_numlist
                             .apply(extract_year_from_title))

    # add year to test dataset
    interim_test["title_numlist"] = (interim_test
                                     .title
                                     .str
                                     .findall(r'\b\d+\b'))

    interim_test["year"] = (interim_test
                            .title_numlist
                            .apply(extract_year_from_title))

    # add is_blend to train dataset
    interim_train["is_blend"] = (interim_train
                                 .variety
                                 .apply(extract_blend_from_variety))

    # add is_blend to test dataset
    interim_test["is_blend"] = (interim_test
                                .variety
                                .apply(extract_blend_from_variety))

    # drop unnecessary columns from train test datasets
    features_added_train = (interim_train
                            .drop(columns=["description",
                                           "title",
                                           "title_numlist"]))

    features_added_test = (interim_test
                           .drop(columns=["description",
                                           "title",
                                           "title_numlist"]))

    _save_datasets(features_added_train, features_added_test, out_dir, flag)

if __name__ == '__main__':
    extract_features()