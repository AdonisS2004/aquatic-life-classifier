################
#   IMPORTS    #
################

import os

from ..tools.data.data_fetch import download_kaggle_datasets
from ..tools.data.data_process import (
    create_data_file_structure, 
    process_collection,
    process_raw
)
from ..tools.file.file_manager import (
    get_git_project_root,
    import_json_to_dict,
)

#################
#   VARIABLES   #
#################

ROOT = get_git_project_root(os.path.dirname(os.path.realpath(__file__)))
unzip_data = True

##################
#   FUNCTIONS    #
##################

def generate_data(data_config_path: str) -> dict:
    """
    Args:
        str: path to data config file
    Returns:
        dict: documentation about image data processing

    Procedure:
        1. (done) Create file structure.
        2. (done) Download data. place in appropriate path (collection). 
        3. Process data.
            a. Place species and appropriate folders species/img.png.
            b. Each image should be named species_name_collection_{id}.
            c. Directory should be raw/species/img.png
        4. For each image:
            a. Resize Image.
            b. Normalize Pixel Values.
        5. Save processed images into processed/species/img.png
        6. Document data.
    """
    # import data config
    data_args = import_json_to_dict(data_config_path)

    # create data path; grab key data paths
    data_folder_path = create_data_file_structure(ROOT, data_args["collections"])
    raw_path = os.path.join(data_folder_path, "raw")
    splits_path = os.path.join(data_folder_path, "splits")
    collections_path = os.path.join(data_folder_path, "collections")

    # download data
    download_kaggle_datasets(data_args["datasets"]["kaggle"], collections_path, unzip_data)

    # process each collection
    for collection in data_args["collections"]:
        process_collection(collection, data_folder_path, raw_path)
    pass


