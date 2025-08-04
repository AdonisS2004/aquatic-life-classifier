###############
#   Imports   #
###############
import os
import logging
from datetime import datetime
from .utils.management.file_manager import get_git_project_root 
from .data_processor.data_service import (
    generate_data,
    process_data
)
from .data_classifier import cnn
from .data_pipeline import pipeline_service as ps

#################
#   VARIABLES   #
#################

# date setup
dt_obj = datetime.now()
formatted_date = dt_obj.strftime("%Y-%m-%d %H:%M:%S")

# path setup
ROOT = get_git_project_root(os.path.dirname(os.path.realpath(__file__)))
DATA_CONFIG = os.path.join(ROOT, "configs", "data_config.json")
path_to_raw = os.path.join(ROOT, "data", "raw")
path_to_processed = os.path.join(ROOT, "data", "processed")

# logging setup
LOG_PATH = os.path.join(ROOT, "logs", f"{formatted_date}.log")
logging.basicConfig(
    filename=LOG_PATH, 
    encoding='utf-8', 
    level=logging.INFO, 
    format='%(asctime)s %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p',
)

##################
#   FUNCTIONS    #
##################

# main function
def main() -> None:
    """
    Main location for executable logic for Classifcaiton model
    """
    # generate_data(DATA_CONFIG)
    # process_data(path_to_raw, path_to_processed)
    ps.setup_training_pipeline()

if __name__ == "__main__":
    main()