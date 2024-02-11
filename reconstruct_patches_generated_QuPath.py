# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 01:27:40 2024

@author: Omnia
"""


import os
import pandas as pd
import numpy as np
from PIL import Image

# Define the path to patches and DataFrame
path_to_patches = r"C:\Users\Omnia\Desktop\data\subset_data\NH19-2429"

# Function to extract spatial information from filename
def extract_spatial_info(filename):
    parts = filename[:-4].split(' ')
    coordinates = parts[1].split('_')
    dimensions_x = coordinates[1].split('-')
    x = int(dimensions_x[1])
    dimensions_y = coordinates[2].split('-')
    y = int(dimensions_y[1])
    dimensions_w = coordinates[3].split('-')
    w = int(dimensions_w[1])
    dimensions_h = coordinates[4].split('-')
    h = int(dimensions_h[1])
    return x, y, w, h

# List image filenames
image_filenames = os.listdir(path_to_patches)

# Sort filenames based on spatial information
sorted_filenames = sorted(image_filenames, key=lambda x: extract_spatial_info(x))

# Create a dictionary to store x, y, w, h information for each filename
spatial_info_dict = {}

# Populate spatial_info_dict with filename and spatial information
for filename in sorted_filenames:
    x, y, w, h = extract_spatial_info(filename)
    spatial_info_dict[filename] = {'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h}

#%%
# Find the maximum dimensions for creating the canvas
max_x = max(spatial_info_dict[filename]['x2'] for filename in spatial_info_dict)
max_y = max(spatial_info_dict[filename]['y2'] for filename in spatial_info_dict)

# Create the canvas with the maximum dimensions
canvas = np.zeros((max_y, max_x, 3), dtype=np.uint8)

# Place each patch at its specified location on the canvas
for filename in spatial_info_dict:
    x1, x2, y1, y2 = spatial_info_dict[filename]['x1'], spatial_info_dict[filename]['x2'], spatial_info_dict[filename]['y1'], spatial_info_dict[filename]['y2']

    patch_image_path = os.path.join(path_to_patches, filename)
    patch_image = Image.open(patch_image_path)

    # Resize the patch image to match specified dimensions
    patch_image = patch_image.resize((x2 - x1, y2 - y1))

    patch_image = np.array(patch_image)

    # Place the patch on the canvas
    canvas[y1:y2, x1:x2, :] = patch_image

reconstructed_image_pil = Image.fromarray(canvas)
reconstructed_image_pil.show()

