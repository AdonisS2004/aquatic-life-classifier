import os
import shutil

# from ..image.image_process import (
#     image_normalize,
#     image_resize
# )

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
            
        Return:
            None
        Reference: https://stackoverflow.com/questions/33135038/how-do-i-use-os-scandir-to-return-direntry-objects-recursively-on-a-directory/33135143
        """
        id_no = 1 # starting ID number for image
        for directory in os.scandir(path):
            if directory.is_dir():
                yield from recur_helper(directory_name + "_" + directory.name, directory.path)
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
            if (file_path == destination_path):
                raise Exception("same destionation: {file_path} : {destination_path}")
            shutil.copy(file_path, destination_path)
        except FileNotFoundError:
            print(f"Error: Source file '{file_path}' not found.")
        except Exception as e:
            print(f"An error occured: {e}")
    

def process_raw():
    # TO-DO
    pass
