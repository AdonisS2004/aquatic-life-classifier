import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from ..utils.decorators import (
    log
)
from ..utils.file_manager import(
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

def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation for better generalization
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='mps', 
                weight_decay=0.0001, scheduler_patience=10, scheduler_factor=0.5, 
                early_stopping_patience=20, use_mixup=True, mixup_alpha=0.2):
    """
    Complete training pipeline with advanced techniques
    """
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = 'models/best_model.pth'
    patience_counter = 0
    
    print("Starting training...")
    print("="*50)
    print(f"Using Mixup: {use_mixup}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print("="*50)
    
    for epoch in range(1, num_epochs + 1):
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, 
            use_mixup=use_mixup, mixup_alpha=mixup_alpha
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
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, best_model_path)
            print(f'New best model saved! Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch} epochs')
            break
            
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