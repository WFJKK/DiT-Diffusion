"""Visualizes the denoising process of a pretrained DiffusionTransformer on validation images."""



import torch
from package.diffuser import Diffuser
from package.model import DiffusionTransformer
from package.sampling import denoise_img_ddpm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from package.data import  val_dataset

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
    img_H=32,          
    img_W=32,
    embed_dim=256,      
    hidden_dim=1024 ,     
    patch_size_H=4,     
    patch_size_W=4,
    num_heads=4,        
    mha_dropout=0.1,
    num_layers=4,       
    dropout=0.1
)
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)


model.eval()
diffuser.eval()

num_images = 5
images = torch.stack([val_dataset[i][0] for i in range(num_images)]).to(device)

t_start = 130
t = torch.full((num_images,), t_start, device=device, dtype=torch.long)


x_t, _ = diffuser(images, t)


with torch.no_grad():
    x0_denoised = denoise_img_ddpm(
        x_t=x_t,
        t_start=t_start,
        model=model,
        alphas=diffuser.alphas,
        betas=diffuser.betas,
        alphas_cumprod=diffuser.alphas_cumprod
    )


for i in range(num_images):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(TF.to_pil_image(images[i].cpu().clamp(0, 1)))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(TF.to_pil_image(x_t[i].cpu().clamp(0, 1)))
    axes[1].set_title(f"Noisy (t={t_start})")
    axes[1].axis("off")

    axes[2].imshow(TF.to_pil_image(x0_denoised[i].cpu().clamp(0, 1)))
    axes[2].set_title("Denoised")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


