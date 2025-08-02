###############
#   Imports   #
###############
import os
from data_processor.utils.tools.file.file_manager import get_git_project_root 
from data_processor.data_service import (
    generate_data,
    process_data
)
#################
#   VARIABLES   #
#################
ROOT = get_git_project_root(os.path.dirname(os.path.realpath(__file__)))
DATA_CONFIG = os.path.join(ROOT, "configs", "data_config.json")

path_to_raw = os.path.join(ROOT, "data", "raw")
path_to_processed = os.path.join(ROOT, "data", "processed")

if __name__ == "__main__":
    generate_data(DATA_CONFIG)
    process_data(path_to_raw, path_to_processed)