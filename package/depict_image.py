"""
Utility function to display normalized image tensors using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    """
    Display a normalized image tensor (C, H, W) as a matplotlib image.

    Args:
        img (Tensor): A torch tensor image normalized to [-1, 1].
    """
    img = img.numpy()               
    img = img.transpose((1, 2, 0))  
    img = img * 0.5 + 0.5           
    img = np.clip(img, 0, 1)        
    plt.imshow(img)
    plt.axis('off')
    plt.show()

