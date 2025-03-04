import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from patchify import patchify

# Step 1: Read the image
image_path = r"C:/Users/omnia/OneDrive - University of Jeddah/PhD progress/PhD_Thesis/Figs/sunset.jpeg"
image = Image.open(image_path).convert('RGB')  # Ensure RGB mode
image_array = np.array(image)

# Step 2: Ensure image has 3 channels (handle grayscale case)
if len(image_array.shape) == 2:  # If grayscale
    image_array = np.expand_dims(image_array, axis=-1)  # Convert to (H, W, 1)

# Step 3: Define Patch Size
patch_height, patch_width = 256, 256  # Patch size
step = patch_height  # Step size should match patch height for non-overlapping patches

# Ensure image dimensions are divisible by patch size
height, width, channels = image_array.shape
if height % patch_height != 0 or width % patch_width != 0:
    pad_height = (patch_height - (height % patch_height)) % patch_height
    pad_width = (patch_width - (width % patch_width)) % patch_width
    
    # Pad image using np.pad
    image_array = np.pad(image_array, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

    print(f"Image padded to shape: {image_array.shape}")

# Step 4: Patchify the image 
patches = patchify(image_array, (patch_height, patch_width, channels), step=step)  # Ensure correct patch size

# Step 5: Remove the extra dimension
patches = np.squeeze(patches)  # Removes singleton dimensions (if any)

# Step 6: Ensure the patches are in the correct shape
print("Patches shape after squeeze:", patches.shape)  # Expecting (num_patches_h, num_patches_w, patch_height, patch_width, num_channels)

# Step 7: Extract dimensions safely
num_patches_h, num_patches_w, patch_height, patch_width, num_channels = patches.shape  # Now 5D

# Step 8: Display patches in a grid
fig, axes = plt.subplots(num_patches_h, num_patches_w, figsize=(10, 10))

for i in range(num_patches_h):
    for j in range(num_patches_w):
        patch = patches[i, j]  # Extract patch
        axes[i, j].imshow(patch)
        axes[i, j].axis("off")  # Hide axes

plt.tight_layout()
plt.show()
