from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ...utils.management.file_manager import iter_files_in_folder

import os
import logging
from PIL import Image

#################
#   Variables   #
#################

logger = logging.getLogger(__name__)

###############
#   Classes   #
###############

def AquaticLifeDataset(Dataset):
    """
    Custom Aquatic Life Dataset
    """
    def __init__(self, data_dir:str, transform = None, phase:str = "train") -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.phase = phase

        self.image_paths = []
        self.image_labels = []
        self.index_to_class = {}
        self.class_to_index = {}

    def _load_dataset(self):
        """ 
        Populates the dataset with data from the data_dir
        """
        n, labels = sorted(get_labels(self.data_dir))
        for idx, label in enumerate(labels):
            image_folder = os.path.join(self.data_dir, label)
            self.index_to_class[idx] = label
            self.class_to_index[label] = idx

            for image_file in iter_files_in_folder(image_folder):
                self.image_paths.append(os.path.join(image_folder, image_file))
                self.image_labels.append(idx) # used to map idx of image path to class
        print(f"Loaded {len(self.image_paths)} images from {len(n)} species")


    def __len__(self) -> int:
        """ Returns the number of classes/labels in the dataset
        Return
            int: number of classes/labels in the dataset
        """
        return len(self.image_labels)

    def __get_item__(self, idx):
        """
        Gets
        Args:
            int: index of the data to get
        Returns:
            tuple(Image, int): Image object for classification and label int
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.exception(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0)) # return black image if exception

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_class_distribution(self):
        """
        Gets the distribution of the data
        """
        class_count = {}
        
        for label in self.image_labels:
            class_name = self.index_to_class[label]
            if class_name not in class_count:
                class_count[class_name] = 0
            class_count[class_name] += 1

        return class_count



#################
#   Functions   #  
#################

def get_labels(path: str) -> tuple[int, list[str]]:
    """ Get the name of the species. Each name will be (source_label)
    Arg:
        str: path to 
    Return
        tuple[int, list[str]]
    """
    res = [str(label) for label in iter_files_in_folder(path) 
                        if os.path.isdir(os.path.join(path, label))]
    n = len(res)
    return (n, res)

def create_data_loaders():
    pass

def create_data_transforms():
    pass

def train_epoch():
    pass

def validate_epoch():
    pass