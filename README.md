# PictureDenoiser

A PyTorch-based image denoising pipeline using a diffusion model trained on CIFAR10 data.
Can also be used for image generation (with a different dataset though).

---

## Features


- **Diffusion-based denoising** implemented from scratch.
- **Patchification** and reconstruction utilities.
- **Training and inference** workflows with logging.
- **Minimal external dependencies**.

---

##  Project Structure

- `/package`: main package files
- `actually_denoising.py`: showing the denoiser in action for specified time step
- `training.py`:training file
- `requirements.txt`: dependencies




---

##  Installation

```bash
git clone https://github.com/WFJKK/DiT-Diffusion.git
cd DiT-Diffusion
pip install -r requirements.txt
```

---

##  Training

```bash
python picturedenoiser/training.py
```

Optional configuration may be done inside `TrainerClass.py` or passed via argparse if implemented.

---

##  Denoising Inference

picturedenoiser/actually_denoising.py



---

##  Dependencies

Listed in `requirements.txt`,includes:

matplotlib
numpy
torch
torchvision

---



---

## License

MIT License 

---

