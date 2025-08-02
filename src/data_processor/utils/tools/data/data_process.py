import os
import shutil
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
    def recur_helper(directory_name: str, path: str):
        """ Generator for Recursively find each species in a folder, assign them names and ID's
        Args:
            str: name to make the directory
            str: current path to file
        Return:
            None
        Reference: https://stackoverflow.com/questions/33135038/how-do-i-use-os-scandir-to-return-direntry-objects-recursively-on-a-directory/33135143
        """
        id_no = 1 # starting ID number for image
        for directory in os.scandir(path):
            if directory.is_dir():
                yield from recur_helper(directory_name + "." + directory.name, directory.path)
            else:  
                yield (directory_name, directory.path, id_no)
                id_no += 1

    path_to_collection = os.path.join(collections_path, collection)

    # check path existence, raise error otherwise
    if not os.path.isdir(collections_path):
        raise FileExistsError(f"Collections path does not exist: {collections_path}")
    if not os.path.isdir(raw_path):
        raise FileExistsError(f"Raw path does not exist: {raw_path}")
    if not os.path.isdir(path_to_collection):
        raise FileExistsError(f"Collection does not exist: {path_to_collection}")

    for (file_name, file_path, id_no) in recur_helper(collection, path_to_collection):
        try:
            destination_folder = os.path.join(raw_path, file_name)
            destination_path = os.path.join(destination_folder, file_name + "_" + f'{id_no}' + ".jpg")
            os.makedirs(destination_folder, exist_ok=True)
            shutil.copy(file_path, destination_path)
        except FileNotFoundError:
            print(f"Error: Source file '{file_path}' not found.")
        except Exception as e:
            print(f"An error occured: {e}")
    

def process_raw_image(path: str, destination: str, new_height: int = 224, new_width: int = 224, lower_norm: float = 0.0, upper_norm = 1.0) -> None:
    """ Resizes and Normalizes all images path. 
    Resulting image size: 224 x 224
    Args:
        str: path that contains all the data to be processed
        str: destination for where new data will be located
    Returns:
        none
    """
    def recur_helper(path: str, destination: str):
        """ Recursively generates image paths to generate
        Args:
        str: current
        str: destination for data
        Returns:
            none
        """
        for file in os.scandir(path):
            if file.is_dir():
                yield from recur_helper(file.path, os.path.join(destination, file.name))
            else:
                yield (file.path, destination, os.path.join(destination, file.name))

    for (image_path, image_folder, image_destination) in recur_helper(path, destination):
        os.makedirs(image_folder, exist_ok=True)
        shutil.copy(image_path, image_destination)
        image_resize(image_path, image_destination, new_height, new_width)
        image_normalize(image_destination,image_destination)