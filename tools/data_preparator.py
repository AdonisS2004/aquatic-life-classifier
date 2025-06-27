import os
import zipfile
from file_manager import get_git_project_root

ROOT = get_git_project_root(os.path.abspath(__file__))
data_path = os.path.join(ROOT, 'data', 'collection')



