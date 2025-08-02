# TO-DO: Add in a logger to replace debug print statements

################
#   IMPORTS    #
################
import os
from .utils.tools.data.data_fetch import download_kaggle_datasets
from .utils.tools.data.data_process import (
    create_data_file_structure, 
    process_collection,
    process_raw_image
)
from .utils.tools.file.file_manager import (
    import_json_to_dict,
    get_git_project_root
)

#################
#   VARIABLES   #
#################

ROOT = get_git_project_root(os.path.dirname(os.path.realpath(__file__)))
unzip_data = True

##################
#   FUNCTIONS    #
##################

def generate_data(data_config_path: str) -> None:
    """
    Args:
        str: path to data config file
    Returns:
        dict: documentation about image data processing
    """

    # import data config
    print(f'importing data_config json to a dictionary')
    data_args = import_json_to_dict(data_config_path)
    print(f'finished importing data_config json')    

    # create data path; grab key data paths
    data_folder_path = create_data_file_structure(ROOT)
    raw_path = os.path.join(data_folder_path, "raw")
    splits_path = os.path.join(data_folder_path, "splits")
    collections_path = os.path.join(data_folder_path, "collections")

    # download data
    print(f'downloading kaggle dataset')
    download_kaggle_datasets(data_args["datasets"]["kaggle"], collections_path, unzip_data)
    print(f'finished downloading')

    # process each collection
    print(f'processing collections')
    for collection in data_args["collections"]:
        print(f'Processing: {collection}')
        process_collection(collection, collections_path, raw_path)
        
    print(f'finished processing collections')

def process_data(path: str, destination: str, new_height: int = 224, new_width: int = 224) -> None:
    """ Function for processing all folders and images in data/raw
    Args:
        str: path that contains all the data to be processed
        str: destination for where new data will be located
    Returns:
        none
    """
    if not os.path.isdir(path):
        raise FileExistsError("Directory for raw images does not exist")
    process_raw_image(path, destination, new_height, new_width)