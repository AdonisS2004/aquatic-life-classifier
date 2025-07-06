import os
from data_processor.utils.tools.file.file_manager import get_git_project_root 
from data_processor.data_service import generate_data

#################
#   VARIABLES   #
#################

ROOT = get_git_project_root(os.path.dirname(os.path.realpath(__file__)))
DATA_CONFIG = os.path.join(ROOT, "configs", "data_config.json")
unzip_data = True

if __name__ == "__main__":
    generate_data(DATA_CONFIG)