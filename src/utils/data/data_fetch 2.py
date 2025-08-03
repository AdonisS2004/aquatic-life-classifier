import kaggle
import os

def download_datasets():
    # TO-DO if there is another source for data
    pass

def download_kaggle_datasets(datasets: list[str], path_dir: str, unzip:bool = True) -> None:
    """ Download all datasets in list of kaggle datasets. Be sure to authenticate kaggle 
    api locally.
    Args:
        list[str]: name of each dataset
        str: path directory
        bool: immediately unzip files or not
    Returns:
        None
    """
    kaggle.api.authenticate()
    for dataset in datasets:
        kaggle.api.dataset_download_files(dataset, path=os.path.join(path_dir,"kaggle"), unzip=unzip)