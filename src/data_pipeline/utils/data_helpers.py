import torch
from torchvision import transforms
from ...utils.file_manager import iter_files_in_folder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

class AquaticLifeDataset(Dataset):
    """
    Custom Aquatic Life Dataset
    """
    def __init__(self, data_dir, transform=None, phase="train") -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.phase = phase

        self.image_paths = []
        self.image_labels = []
        self.index_to_class = {}
        self.class_to_index = {}

        self._load_dataset()

    def _load_dataset(self):
        """ 
        Populates the dataset with data from the data_dir
        """
        n, labels = get_labels(self.data_dir)
        labels = sorted(labels)
        for idx, label in enumerate(labels):
            image_folder = os.path.join(self.data_dir, label)
            self.index_to_class[idx] = label
            self.class_to_index[label] = idx

            for image_file in iter_files_in_folder(image_folder):
                self.image_paths.append(os.path.join(image_folder, image_file))
                self.image_labels.append(idx) # used to map idx of image path to class
        print(f"Loaded {len(self.image_paths)} images from {n} species")


    def __len__(self):
        """ Returns the number of classes/labels in the dataset
        Return
            int: number of classes/labels in the dataset
        """
        return len(self.image_labels)

    def __getitem__(self, idx):
        """
        Gets
        Args:
            int: index of the data to get
        Returns:
            tuple(Image, int): Image object for classification and label int
        """
        image_path = self.image_paths[idx]
        label = self.image_labels[idx]

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

    # create splits (delete existing spltis and recreate them)
    splits = ["train", "test", "val"]
    for split in splits:
        directory_path = os.path.join(target, split)
        try:
            shutil.rmtree(directory_path)
            os.makedirs(directory_path, exist_ok=True)
        except OSError as e:
            print(f"Error removing directory: {e}")
    
    # split data
    split_info = {"train":[], "test":[], "val":[]}
    for species_class in iter_files_in_folder(source):
        # create/gather directories
        data_source = os.path.join(source, species_class)
        for split in splits:
            directory = os.path.join(target, split, species_class)
            os.makedirs(directory, exist_ok=True)

        # gather all images
        data_files = [image for image in iter_files_in_folder(data_source)]

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
    with open(os.path.join(target,'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nSplit information saved to: {os.path.join(target,'split_info.json')}")
    return split_info  

def create_data_loaders(data_dir:str, batch_size:int=32, num_workers:int=4) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create data loaders for training, validation, and testing
    """
    
    train_transforms, val_transforms = create_data_transforms()
    
    # Create datasets
    train_dataset = AquaticLifeDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transforms,
        phase='train',
    )
    
    val_dataset = AquaticLifeDataset(
        os.path.join(data_dir, 'val'), 
        val_transforms,
        'val',
    )
    
    test_dataset = AquaticLifeDataset(
        os.path.join(data_dir, 'test'),
        val_transforms,
        'test',
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
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Testing samples: {len(test_dataset)}")
    
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

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train model for one epoch
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        optimizer.step()
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        # Update progress bar
        accuracy = 100 * correct_predictions / total_samples
        progress_bar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct_predictions / total_samples
    
    return epoch_loss, epoch_accuracy

def validate_epoch(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_accuracy = 100 * correct_predictions / total_samples
    
    return epoch_loss, epoch_accuracy