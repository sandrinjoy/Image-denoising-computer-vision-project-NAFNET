import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from dataset import get_dataloader
from basicsr.models.archs.Baseline_arch import Baseline

def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu'):
    """
    Train the model
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create directory for saving models
    os.makedirs('checkpoints', exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for noisy_imgs, clean_imgs in train_bar:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': train_loss / len(train_loader)})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_bar:
                noisy_imgs = noisy_imgs.to(device)
                clean_imgs = clean_imgs.to(device)
                
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                val_loss += loss.item()
                val_bar.set_postfix({'loss': val_loss / len(val_loader)})
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, '
              f'Val Loss = {val_loss/len(val_loader):.4f}')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = Baseline(
        width=64,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2]
    ).to(device)
    
    # Get dataloaders
    print('Loading dataset...')
    train_loader = get_dataloader(batch_size=32, num_workers=4, train=True)
    val_loader = get_dataloader(batch_size=32, num_workers=4, train=False)
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=10, device=device)

if __name__ == '__main__':
    main() 