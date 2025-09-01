###############
#   Imports   #
###############
import os
import logging
from datetime import datetime
from .utils.file_manager import get_git_project_root 
from .data_classifier.cnn import (
    create_model,
)
from .data_processor.data_service import (
    generate_data,
    process_data,
)
from .data_pipeline.pipeline_service import (
    create_data_splits,
    create_data_loaders,
    train_model,
)
from .utils.file_manager import (
    import_json_to_dict
)

#################
#   VARIABLES   #
#################

# date setup
dt_obj = datetime.now()
formatted_date = dt_obj.strftime("%Y-%m-%d %H:%M:%S")

# path setup
ROOT = get_git_project_root(os.path.dirname(os.path.realpath(__file__)))
DATA_CONFIG = os.path.join(ROOT, "configs", "data_config.json")
TRAINING_CONFIG = os.path.join(ROOT, "configs", "training_config.json")
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
logger = logging.getLogger(__name__)

##################
#   FUNCTIONS    #
##################

# main function
def main() -> None:
    """
    Main location for executable logic for Classification model training
    """
    training_config = import_json_to_dict(TRAINING_CONFIG)
    source_data_dir = os.path.join(ROOT,'data','processed')
    split_data_dir = os.path.join(ROOT,'data','splits')

    # Step 1: Generate data 
    # generate_data(ROOT, DATA_CONFIG)

    # Step 2: Process the Data
    process_data(path_to_raw, path_to_processed)

    # Step 3: Split Data
    logger.info(f"Device: {training_config['device']}")
    logger.info(f"Batch size: {training_config['batch_size']}")
    logger.info(f"Learning rate: {training_config['learning_rate']}")
    logger.info(f"Weight decay: {training_config.get('weight_decay', 0.0001)}")
    logger.info(f"Using weighted sampling: {training_config.get('use_weighted_sampling', True)}")
    logger.info(f"Using mixup: {training_config.get('use_mixup', True)}")
    create_data_splits(source_data_dir, split_data_dir)

    # Step 4: Create Data Loaders
    train_loader, val_loader, test_loader, class_to_idx = create_data_loaders(
        split_data_dir, 
        batch_size=training_config['batch_size'],
        use_weighted_sampling=training_config.get('use_weighted_sampling', True)
    )

    # Step 5: Initialize Model
    logger.info("Step 5: Initialize Model")
    model = create_model(num_classes=23, device=training_config['device'])
    logger.info("Step 5: SUCCESS")

    # Step 6: Train Model
    logger.info("Step 6: Train Model")
    train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=training_config['num_epochs'], 
        learning_rate=float(training_config['learning_rate']),
        device=training_config['device'],
        weight_decay=training_config.get('weight_decay', 0.0001),
        scheduler_patience=training_config.get('scheduler_patience', 10),
        scheduler_factor=training_config.get('scheduler_factor', 0.5),
        early_stopping_patience=training_config.get('early_stopping_patience', 20),
        use_mixup=training_config.get('use_mixup', True),
        mixup_alpha=training_config.get('mixup_alpha', 0.2)
    )
    logger.info("Step 6: SUCCESS")

if __name__ == "__main__":
    main()