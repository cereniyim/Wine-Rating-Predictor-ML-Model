from pathlib import Path

import click
import logging
import urllib.request


logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--name')
@click.option('--url')
@click.option('--out-dir')
def download_data(name, url, out_dir):
    """Download a csv file and save it to local disk.

    Parameters
    ----------
    name: str
        name of the csv file on local disk, without '.csv' suffix.
    url: str
        remote url of the csv file.
    out_dir:
        directory where file should be saved to.

    Returns
    -------
    None
    """
    log = logging.getLogger('download-data')
    assert '.csv' not in name, f'Received {name}! ' \
        f'Please provide name without csv suffix'

    out_path = Path(out_dir) / f'{name}.csv'

    log.info('Downloading dataset')
    log.info(f'Will write to {out_path}')

    urllib.request.urlretrieve(url, out_path)


if __name__ == '__main__':
    download_data()
