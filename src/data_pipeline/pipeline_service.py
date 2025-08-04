import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from ..utils.management.decorators import (
    log
)
from ..utils.management.file_manager import(
    get_git_project_root,
)
from .utils.data_helpers import (
    train_epoch,
    validate_epoch,
    create_data_splits,
    create_data_loaders,
    create_data_transforms,
)

from ..data_classifier import cnn

#################
#   Variables   #
#################

ROOT = get_git_project_root(os.path.dirname(os.path.realpath(__file__)))
logger = logging.getLogger(__name__)

#################
#   Functions   #  
#################

@log(include_timer=True)
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='mps'):
    """
    Complete training pipeline
    """
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = 'best_model.pth'
    
    print("Starting training...")
    print("="*50)
    
    for epoch in range(1, num_epochs + 1):
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f'Epoch [{epoch}/{num_epochs}]:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, best_model_path)
            print(f'New best model saved! Val Acc: {val_acc:.2f}%')
        
        print()
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return history, best_model_path

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation metrics
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', color='blue')
    ax2.plot(history['val_acc'], label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

@log(include_timer=True)
def setup_training_pipeline():
    """
    Complete setup example for your sea life dataset
    """
    
    print("Aquatic Life CNN Training Pipeline Setup")
    print("="*50)
    
    # Configuration
    config = {
        'source_data_dir': os.path.join(ROOT,'data','processed'),  # Your original species folders
        'split_data_dir': os.path.join(ROOT,'data','splits'),  # Where to create train/val/test
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'device': 'mps' if torch.mps.is_available() else 'cpu'
    }
    
    logger.info(f"Device: {config['device']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    
    # Step 1: Create data splits (run once)
    logger.info("Step 1: Creating data splits...")
    create_data_splits(config['source_data_dir'], config['split_data_dir'])
    
    # Step 2: Create data loaders
    logger.info("Step 2: Creating data loaders...")
    train_loader, val_loader, test_loader, class_to_idx = create_data_loaders(
        config['split_data_dir'], 
        batch_size=config['batch_size']
    )
    
    # Step 3: Initialize model
    logger.info("Step 3: Initializing model...")
    model = cnn.create_model(num_classes=46, device=config['device'])
    
    # Step 4: Start training
    logger.info("Step 4: Ready to start training!")
    train_model(model, train_loader, val_loader)

    logger.info("Training Complete")
    
    return config

