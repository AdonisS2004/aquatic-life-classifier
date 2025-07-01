import kaggle

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
        kaggle.api.dataset_download_files(dataset, path=path_dir, unzip=unzip)