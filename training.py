"""Training script for a DiffusionTransformer model using MSE loss and dataset-based validation."""



import torch 
import os
from package.TrainerClass import DiffusionTransformerTrainer
from package.data import train_dataset, val_dataset
from package.diffuser import Diffuser 
from package.model import DiffusionTransformer
import torch.optim as optim

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA (GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

n_diffusion_steps = 1000
diffuser = Diffuser(timesteps=n_diffusion_steps).to(device)

model = DiffusionTransformer(
    n_diffusion_steps=n_diffusion_steps,
    input_dim=3,        
    img_W=32,
    embed_dim=256,      
    hidden_dim=1024 ,    
    patch_size_H=4,     
    patch_size_W=4,
    num_heads=4,        
    mha_dropout=0.1,
    num_layers=4,       
    dropout=0.1
).to(device)



trainer = DiffusionTransformerTrainer(model, diffuser, train_dataset, val_dataset)
trainer.to(device)

batch_size = 64
epochs = 1000
lr = 1e-4

train_loader, val_loader = trainer.get_dataloaders(batch_size)
optimizer = optim.Adam(model.parameters(), lr=lr)



best_val_loss = float('inf')  
save_path_best = 'best_model.pth'
save_path_checkpoint = 'checkpoint.pth'


if os.path.exists(save_path_best):
    model.load_state_dict(torch.load(save_path_best))
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_data in val_loader:
            images_val, labels = batch_data
            images_val = images_val.to(device)
            loss_val = trainer.calc_loss(images_val)
            running_val_loss += loss_val.item()
    best_val_loss = running_val_loss / len(val_loader)
    print(f"Resumed from best model with val loss {best_val_loss:.4f}")


for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1} training starts")

    for batch_data in train_loader:
        images, labels = batch_data
        images = images.to(device)

        optimizer.zero_grad()
        loss = trainer.calc_loss(images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"The training loss is {avg_train_loss:.4f}")

    
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_data in val_loader:
            images_val, labels = batch_data
            images_val = images_val.to(device)
            loss_val = trainer.calc_loss(images_val)
            running_val_loss += loss_val.item()

    avg_val_loss = running_val_loss / len(val_loader)
    print(f"The validation loss is {avg_val_loss:.4f}")

    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path_best)
        print(f"Saved best model with val loss {best_val_loss:.4f}")

