import numpy as np
import matplotlib.pyplot as plt

    
def plot_mask(img, mask):
    mask = np.copy(mask)
    for j in range(5):
        mask[j, 0] = j + 1

    plt.imshow(img, cmap='gray')
    mask = np.where(mask, mask, np.nan)
    plt.imshow(mask, cmap='Set3', alpha=0.3)        
    plt.axis(False)

    
def show_cmap():
    labels = {1: "liver", 2: "spleen", 3: "left-kidney", 4: "right-kidney", 5: "bowel"}

    plt.imshow(np.arange(1, 6)[None], cmap='Set3', alpha=0.5)  

    for i in range(len(labels)):
        plt.text(
            i, 0, labels[i + 1],
            horizontalalignment='center',
            verticalalignment='center',
        )

    plt.axis(False)
    plt.show()