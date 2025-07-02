import os

from ..file.file_manager import get_git_project_root
from ..image.image_process import (
    image_normalize,
    image_resize
)

def create_data_file_structure(path_dir: str) -> str:
    """ Recursively creates data file structure for ML purposes
    Args:
        str: the path directory where folders should be created
        list[str]: path of collections to create 
    Returns:
        str: path to data folder
    """
    folders = [
        "data/collections/",
        "data/processed",
        "data/raw",
        "data/splits/train"
        "data/splits/test"
        "data/splits/val"
    ]

    for folder in folders:
        directory = os.path.join(path_dir, folder)
        os.makedirs(directory , exist_ok=True)

    return os.path.join(path_dir, "data") 

def process_collection(collection: str, collections_path:str, raw_path: str) -> None:
    """ Processes collection in the collecitons folder. Moves to raw data path
    Args:
        str: name of collection
        str: path of data folder
        str: path of where raw data should go
    Return:
        None
    """
    collection_path = os.path.join(collections_path, collection)

    # check path existence, raise error otherwise
    if not os.path.isdir(collections_path):
        raise FileExistsError("Collections path does not exist.")
    if not os.path.isdir(collection_path):
        raise FileExistsError("Collection does not exist.")

    """
    Procedure:
        1. For each dataset in collection folder:
            a. For each folder in dataset
                i. extract folder name -> (species)
                ii. create new folder in raw path named {species} (or check for existence)
                iii. for each image inside of folder
                    1. copy file into raw path folder named {species}
                    2. rename file species_name_collection_{no.}
    """
    pass


def process_raw():
    # TO-DO
    pass