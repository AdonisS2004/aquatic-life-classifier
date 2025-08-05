################
#   IMPORTS    #
################
import os
import logging
from .utils.data_fetch import download_kaggle_datasets
from .utils.data_process import (
    create_data_file_structure, 
    process_collection,
    process_raw_image
)
from ..utils.file_manager import (
    import_json_to_dict,
)
from ..utils.decorators import (
    log
)

#################
#   VARIABLES   #
#################

logger = logging.getLogger(__name__)
unzip_data = True

##################
#   FUNCTIONS    #
##################

@log(include_timer=True)
def generate_data(root_path: str, data_config_path: str) -> None:
    """
    Args:
        str: path to data config file
    Returns:
        dict: documentation about image data processing
    """

    # import data config
    logger.info(f'Importing data_config json to a dictionary')
    data_args = import_json_to_dict(data_config_path)
    logger.info(f'Finished importing data_config.json')    

    # create data path; grab key data paths
    data_folder_path = create_data_file_structure(root_path)
    raw_path = os.path.join(data_folder_path, "raw")
    collections_path = os.path.join(data_folder_path, "collections")

    # download kaggle data
    logger.info(f'Downloading kaggle dataset')
    download_kaggle_datasets(data_args["datasets"]["kaggle"], collections_path, unzip_data)
    logger.info(f'Finished downloading kaggle dataset')

    # process each collection
    logger.info(f'Processing collections')
    for collection in data_args["collections"]:
        logger.info(f'Processing: {collection}')
        process_collection(collection, collections_path, raw_path)
        logger.info(f'Collection Processed: {collection}')
    logger.info(f'Finished processing collections')

@log(include_timer=True)
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
    logger.info("Processing raw data")
    process_raw_image(path, destination, new_height, new_width)
    logger.info("Finished Processing raw data")