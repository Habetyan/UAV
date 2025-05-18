import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import logging
from datetime import datetime
import seaborn as sns  # Import seaborn explicitly
import os

from model import EnhancedBatteryPredictor
from dataset import prepare_enhanced_data

# Initialize gradient scaler for mixed precision training
scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None, scaler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with tqdm(train_loader, desc='Training', ncols=100) as pbar:
        for batch_idx, (X_batch, y_batch) in enumerate(pbar):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1)  # Reshape target to match output
            
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=True) if device.type == 'cuda' else torch.no_grad():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            if scaler is not None:
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular backward pass without mixed precision
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Update learning rate if using OneCycleLR
            if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
            # Update progress bar with running averages
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'last_loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    rmse = np.sqrt(mean_squared_error(np.array(all_targets).reshape(-1), np.array(all_preds).reshape(-1)))
    r2 = r2_score(np.array(all_targets).reshape(-1), np.array(all_preds).reshape(-1))
    
    return avg_loss, rmse, r2

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True) if device.type == 'cuda' else torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).view(-1, 1)  # Reshape target to match output
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    rmse = np.sqrt(mean_squared_error(np.array(all_targets).reshape(-1), np.array(all_preds).reshape(-1)))
    r2 = r2_score(np.array(all_targets).reshape(-1), np.array(all_preds).reshape(-1))
    
    return avg_loss, rmse, r2

def plot_metrics(metrics, save_path='training_metrics.png'):
    sns.set_style("whitegrid")  # Use seaborn's whitegrid style
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Plot losses
    axes[0,0].plot(metrics['train_loss'], label='Train')
    axes[0,0].plot(metrics['val_loss'], label='Validation')
    axes[0,0].set_title('Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MSE Loss')
    axes[0,0].legend()
    
    # Plot RMSE
    axes[0,1].plot(metrics['train_rmse'], label='Train')
    axes[0,1].plot(metrics['val_rmse'], label='Validation')
    axes[0,1].set_title('RMSE')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('RMSE')
    axes[0,1].legend()
    
    # Plot R²
    axes[1,0].plot(metrics['train_r2'], label='Train')
    axes[1,0].plot(metrics['val_r2'], label='Validation')
    axes[1,0].set_title('R² Score')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('R²')
    axes[1,0].legend()
    
    # Plot Learning Rate
    axes[1,1].plot(metrics['learning_rates'])
    axes[1,1].set_title('Learning Rate')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Learning Rate')
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Configuration
    config = {
        'data_dir': 'BED',
        'batch_size': 64,
        'seq_length': 32,
        'hidden_size': 192,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'epochs': 50,
        'patience': 7,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'use_mixed_precision': torch.cuda.is_available()  # Only use mixed precision with CUDA
    }
    
    logging.info(f"Using device: {config['device']}")
    logging.info("Configuration: " + str(config))
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Prepare data
    train_loader, val_loader, test_loader, input_size = prepare_enhanced_data(
        config['data_dir'],
        seq_length=config['seq_length'],
        batch_size=config['batch_size']
    )
    
    # Initialize model
    model = EnhancedBatteryPredictor(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(config['device'])
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler - now using total steps
    total_steps = len(train_loader) * config['epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Initialize gradient scaler for mixed precision training if CUDA is available
    scaler = torch.amp.GradScaler('cuda') if config['use_mixed_precision'] else None
    
    # Training tracking
    best_val_loss = float('inf')
    no_improve_count = 0
    metrics = {
        'train_loss': [], 'val_loss': [],
        'train_rmse': [], 'val_rmse': [],
        'train_r2': [], 'val_r2': [],
        'learning_rates': []
    }
    
    # Training loop
    start_time = time.time()
    try:
        for epoch in range(1, config['epochs'] + 1):
            epoch_start = time.time()
            logging.info(f"\nEpoch {epoch}/{config['epochs']}")
            
            # Train
            train_loss, train_rmse, train_r2 = train_epoch(
                model, train_loader, criterion, optimizer, 
                config['device'], scheduler, scaler
            )
            
            # Validate
            val_loss, val_rmse, val_r2 = validate(
                model, val_loader, criterion, config['device']
            )
            
            # Update metrics
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)
            metrics['train_rmse'].append(train_rmse)
            metrics['val_rmse'].append(val_rmse)
            metrics['train_r2'].append(train_r2)
            metrics['val_r2'].append(val_r2)
            metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            logging.info(
                f"Train Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}\n"
                f"Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}\n"
                f"Epoch time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'config': config
                }, 'best_model.pt')
                logging.info(f"Saved new best model with validation loss: {val_loss:.6f}")
            else:
                no_improve_count += 1
                if no_improve_count >= config['patience']:
                    logging.info(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Plot current metrics
            plot_metrics(metrics)
            
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
    finally:
        # Load best model and evaluate on test set
        if os.path.exists('best_model.pt'):
            checkpoint = torch.load('best_model.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            test_loss, test_rmse, test_r2 = validate(model, test_loader, criterion, config['device'])
            
            # Log final results
            total_time = time.time() - start_time
            logging.info("\nTraining completed!")
            logging.info(f"Total training time: {total_time/60:.2f} minutes")
            logging.info("\nTest Set Results:")
            logging.info(f"Loss: {test_loss:.4f}")
            logging.info(f"RMSE: {test_rmse:.4f}")
            logging.info(f"R²: {test_r2:.4f}")
            
            # Save final metrics plot
            plot_metrics(metrics, 'final_training_metrics.png')

if __name__ == "__main__":
    main()  