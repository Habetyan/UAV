import torch
import numpy as np
from model import EnhancedBatteryPredictor
from dataset import prepare_enhanced_data
import logging
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.serialization import add_safe_globals
import pickle

# Add numpy scalar to safe globals
add_safe_globals(['numpy._core.multiarray.scalar'])

def load_best_model(model_path='best_model.pt', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint with weights_only=False for compatibility
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        logging.warning(f"Failed to load with weights_only=False, trying alternative loading method...")
        try:
            # Try loading with pickle support
            checkpoint = torch.load(model_path, map_location=device, pickle_module=pickle)
        except Exception as e2:
            logging.error(f"Both loading attempts failed. Error: {str(e2)}")
            raise
    
    config = checkpoint['config']
    
    # Initialize model
    model = EnhancedBatteryPredictor(
        input_size=checkpoint['model_state_dict']['feature_net.0.weight'].shape[1],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Load state
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            
            all_preds.extend(outputs.cpu().numpy().reshape(-1))
            all_targets.extend(y_batch.cpu().numpy().reshape(-1))
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    
    return {
        'predictions': all_preds,
        'targets': all_targets,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def plot_predictions(results, save_path='prediction_analysis.png'):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot of predictions vs actual
    axes[0,0].scatter(results['targets'], results['predictions'], alpha=0.5)
    axes[0,0].plot([min(results['targets']), max(results['targets'])], 
                   [min(results['targets']), max(results['targets'])], 'r--')
    axes[0,0].set_xlabel('Actual Values')
    axes[0,0].set_ylabel('Predicted Values')
    axes[0,0].set_title('Predictions vs Actual')
    
    # Residuals plot
    residuals = results['predictions'] - results['targets']
    axes[0,1].scatter(results['predictions'], residuals, alpha=0.5)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Predicted Values')
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].set_title('Residuals vs Predicted')
    
    # Distribution of residuals
    sns.histplot(residuals, kde=True, ax=axes[1,0])
    axes[1,0].set_title('Distribution of Residuals')
    axes[1,0].set_xlabel('Residual Value')
    
    # Error metrics
    axes[1,1].text(0.1, 0.7, f"MSE: {results['mse']:.4f}", fontsize=12)
    axes[1,1].text(0.1, 0.5, f"RMSE: {results['rmse']:.4f}", fontsize=12)
    axes[1,1].text(0.1, 0.3, f"R²: {results['r2']:.4f}", fontsize=12)
    axes[1,1].set_title('Error Metrics')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model
    try:
        model, config = load_best_model(device=device)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return
    
    # Prepare test data
    _, _, test_loader, _ = prepare_enhanced_data(
        config['data_dir'],
        seq_length=config['seq_length'],
        batch_size=config['batch_size']
    )
    
    # Evaluate model
    logging.info("Evaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    # Log results
    logging.info("\nTest Set Results:")
    logging.info(f"MSE: {results['mse']:.4f}")
    logging.info(f"RMSE: {results['rmse']:.4f}")
    logging.info(f"R²: {results['r2']:.4f}")
    
    # Plot results
    logging.info("Generating plots...")
    plot_predictions(results)
    logging.info("Results saved to prediction_analysis.png")

if __name__ == "__main__":
    main() 