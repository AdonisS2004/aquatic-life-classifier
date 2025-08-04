from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ...utils.management.file_manager import iter_files_in_folder

import os
import logging
import random
import shutil
import json
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

def create_data_splits(source:str, target:str, train_ratio:float = 0.7, val_ratio:float = 0.15, test_ratio:float = 0.15) -> dict[str:list[tuple[str, int]]]:
    """ Splits existing data into 
    Args:
        str: source path (foders of different species)
        str: target path
        float: ratio of training data (default=0.7)
        float: ratio of validation data (default=0.15)
        float: ratio of test data (default=0.15)
    Return:
        dict[str:list[tuple[str, int]]]: split info dictionary
    """
    # make sure that the ratios are okay
    assert (1.0 - (train_ratio + val_ratio + test_ratio)) < 1e-6

    # create splits (in case they don't exist)
    splits = ["train", "test", "val"]
    for split in splits:
        directory = os.path.join(source, split)
        os.makedirs(directory, exist_ok=True)
    
    # split data
    split_info = {"train":[], "test":[], "val":[]}
    for species_class in iter_files_in_folder(source):
        # create/gather directories
        data_source = os.path.join(source, species_class)
        for split in splits:
            directory = os.path.join(target, split, species_class)
            os.makedirs(directory, exist_ok=True)
        train_path = os.path.join(target, "train", species_class)
        val_path = os.path.join(target, "val", species_class)
        test_path = os.path.join(target, "test", species_class)

        # gather all images
        data_files = [os.path.join(data_source, image) for image in iter_files_in_folder(data_source)]

        # check if no images
        n = len(data_files)
        if len(data_files) == 0:
            print(f"Data Source: {data_source}, is empty.")
            continue
            
        # get random splits
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        random.shuffle(data_files)

        train_files = data_files[:n_train]
        val_files = data_files[n_train:n_train+n_val]
        test_files = data_files[n_train+n_val:]

        # copy over files
        for files, split in [(train_files, "train"), (val_files, "val"), (test_files, "test")]:
            for file in files:
                data_file_source = os.path.join(data_source, file)
                data_file_dest = os.path.join(target, split, species_class, file)
                shutil.copy2(data_file_source, data_file_dest)

        split_info['train'].append((species_class, len(train_files)))
        split_info['val'].append((species_class, len(val_files)))
        split_info['test'].append((species_class, len(test_files)))
    
    # Print split summary
    print("\n" + "="*50)
    print("DATA SPLIT SUMMARY")
    print("="*50)
    
    for split in splits:
        total_images = sum([count for _, count in split_info[split]])
        print(f"{split.upper()}: {total_images} images")
    
    # Save split information
    with open(target / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nSplit information saved to: {target / 'split_info.json'}")
    return split_info  

def create_data_loaders(data_dir:str, batch_size:int=32, num_workers:int=4):
    """
    Create data loaders for training, validation, and testing
    """
    
    train_transforms, val_transforms = create_data_transforms()
    
    # Create datasets
    train_dataset = AquaticLifeDataset(
        data_dir=os.path.join(data_dir, 'train'), 
        transform=train_transforms,
        phase='train'
    )
    
    val_dataset = AquaticLifeDataset(
        data_dir=os.path.join(data_dir, 'val'), 
        transform=val_transforms,
        phase='val'
    )
    
    test_dataset = AquaticLifeDataset(
        data_dir=os.path.join(data_dir, 'test'),
        transform=val_transforms,
        phase='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_index

def create_data_transforms():
    """ Creates data transformers for training and validation/testing
    Args:
        None
    Return:
        training_transformer and validation_transformer
    """
    # Training transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # Slightly larger than target
        transforms.RandomCrop(224),     # Random crop to 224x224
        transforms.RandomHorizontalFlip(p=0.5),  # Fish can face either direction
        transforms.RandomRotation(degrees=15),    # Slight rotation
        transforms.ColorJitter(
            brightness=0.2,    # Underwater lighting varies
            contrast=0.2,      # Different water clarity
            saturation=0.2,    # Color variation underwater
            hue=0.1           # Slight color shifts
        ),
        transforms.RandomAffine(
            degrees=0,
            scale=(0.9, 1.1),  # Slight zoom in/out
            shear=5            # Small perspective changes
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation/test transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transforms, val_transforms

def train_epoch():
    """
    (Insert Description)
    """
    pass

def validate_epoch():
    """
    (Insert Description)
    """
    pass