# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GrazDataset
from model import SemanticDroneANN
from config import DEVICE, BEST_MODEL_PATH

def train_model(train_dir, mask_dir, val_dir, val_mask_dir, class_dict_path, epochs=20, batch_size=8):
    train_dataset = GrazDataset(train_dir, mask_dir, class_dict_path, is_train=True)
    val_dataset = GrazDataset(val_dir, val_mask_dir, class_dict_path, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = SemanticDroneANN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("=> Saved best_model.pth")

if __name__ == "__main__":
    train_img_dir = "clean_dataset/train/images"
    train_mask_dir = "clean_dataset/train/masks"
    val_img_dir = "clean_dataset/val/images"
    val_mask_dir = "clean_dataset/val/masks"
    
    # --- FIX: Point to your class dictionary ---
    dict_path = "training_set/gt/semantic/class_dict.csv" 

    train_model(
        train_dir=train_img_dir, 
        mask_dir=train_mask_dir, 
        val_dir=val_img_dir, 
        val_mask_dir=val_mask_dir,
        class_dict_path=dict_path,
        epochs=20
    )