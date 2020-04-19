import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return 'code-challenge/download-data:0.1'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2019/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )


class MakeDatasets(DockerTask):
    """Task to separate training and test sets"""

    in_csv = luigi.Parameter(default='/usr/share/data/raw/wine_dataset.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/interim/')
    flag = luigi.Parameter('.SUCCESS_MakeDatasets')

    @property
    def image(self):
        return f'code-challenge/make-dataset:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        return [
            'python', 'make_dataset.py',
            '--in-csv', self.in_csv,
            '--out-dir', self.out_dir,
            '--flag', self.flag
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / self.flag)
        )

class CleanData(DockerTask):
    """Task to clean training and test datasets"""

    in_path = '/usr/share/data/interim/'

    in_train_csv = luigi.Parameter(default= in_path + 'train.csv')
    in_test_csv = luigi.Parameter(default= in_path + 'test.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/interim/')
    flag = luigi.Parameter('.SUCCESS_CleanData')

    @property
    def image(self):
        return f'code-challenge/clean-data:{VERSION}'

    def requires(self):
        return MakeDatasets()

    @property
    def command(self):
        return [
            'python', 'clean_data.py',
            '--in-train-csv', self.in_train_csv,
            '--in-test-csv', self.in_test_csv,
            '--out-dir', self.out_dir,
            '--flag', self.flag
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / self.flag)
        )

class ExtractFeatures(DockerTask):
    """Task to extract features from training and test datasets"""

    in_path = '/usr/share/data/interim/'

    in_train_csv = luigi.Parameter(default= in_path + 'train_cleaned.csv')
    in_test_csv = luigi.Parameter(default= in_path + 'test_cleaned.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/interim/')
    flag = luigi.Parameter('.SUCCESS_ExtractFeatures')

    @property
    def image(self):
        return f'code-challenge/extract-features:{VERSION}'

    def requires(self):
        return CleanData()

    @property
    def command(self):
        return [
            'python', 'extract_features.py',
            '--in-train-csv', self.in_train_csv,
            '--in-test-csv', self.in_test_csv,
            '--out-dir', self.out_dir,
            '--flag', self.flag
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / self.flag)
        )

class TransformData(DockerTask):
    """Task to transform categorical features of training and test datasets"""

    in_path = '/usr/share/data/interim/'

    in_train_csv = luigi.Parameter(default= in_path + 'train_features_extracted.csv')
    in_test_csv = luigi.Parameter(default= in_path + 'test_features_extracted.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/interim/')
    flag = luigi.Parameter('.SUCCESS_TransformData')

    @property
    def image(self):
        return f'code-challenge/transform-data:{VERSION}'

    def requires(self):
        return ExtractFeatures()

    @property
    def command(self):
        return [
            'python', 'transform_data.py',
            '--in-train-csv', self.in_train_csv,
            '--in-test-csv', self.in_test_csv,
            '--out-dir', self.out_dir,
            '--flag', self.flag
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / self.flag)
        )

class ImputeData(DockerTask):
    """Task to impute training and test datasets,
    creates features and target separated train and test sets"""

    in_path = '/usr/share/data/interim/'

    in_train_csv = luigi.Parameter(default= in_path + 'train_transformed.csv')
    in_test_csv = luigi.Parameter(default= in_path + 'test_transformed.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/interim/')
    flag = luigi.Parameter('.SUCCESS_ImputeData')

    @property
    def image(self):
        return f'code-challenge/impute-data:{VERSION}'

    def requires(self):
        return TransformData()

    @property
    def command(self):
        return [
            'python', 'impute_data.py',
            '--in-train-csv', self.in_train_csv,
            '--in-test-csv', self.in_test_csv,
            '--out-dir', self.out_dir,
            '--flag', self.flag
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / self.flag)
        )

class TrainModel(DockerTask):
    """Task to train random forest regressor with training datasets"""

    in_path = '/usr/share/data/interim/'

    in_train_features_csv = luigi.Parameter(default= in_path + 'train_features.csv')
    in_train_target_csv = luigi.Parameter(default= in_path + 'train_target.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/output/')

    @property
    def image(self):
        return f'code-challenge/train-model:{VERSION}'

    def requires(self):
        return ImputeData()

    @property
    def command(self):
        return [
            'python', 'train_model.py',
            '--in-train-features-csv', self.in_train_features_csv,
            '--in-train-target-csv', self.in_train_target_csv,
            '--out-dir', self.out_dir
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / 'model.sav')
        )

class EvaluateModel(DockerTask):
    """Task to evaluate random forest regressor with test datasets,
    returns mean square error of the predictions and several plots"""

    in_path = '/usr/share/data/interim/'

    in_test_features_csv = luigi.Parameter(default= in_path + 'test_features.csv')
    in_test_target_csv = luigi.Parameter(default= in_path + 'test_target.csv')
    in_trained_model = luigi.Parameter(default= '/usr/share/data/output/model.sav')
    out_dir = luigi.Parameter(default='/usr/share/data/output/')
    flag = luigi.Parameter('.SUCCESS_EvaluateModel')

    @property
    def image(self):
        return f'code-challenge/evaluate-model:{VERSION}'

    def requires(self):
        return TrainModel()

    @property
    def command(self):
        return [
            'python', 'evaluate_model.py',
            '--in-test-features-csv', self.in_test_features_csv,
            '--in-test-target-csv', self.in_test_target_csv,
            '--in-trained-model', self.in_trained_model,
            '--out-dir', self.out_dir,
            '--flag', self.flag
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / self.flag)
        )