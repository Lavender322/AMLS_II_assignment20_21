import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_image(path):
    """Loads an LR image."""
    return np.array(Image.open(path))


def plot_sample(lr, sr):
    """Plots an LR image and its super-resolved version side by side."""
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])