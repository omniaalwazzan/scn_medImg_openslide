directory_path = r'D:\_Guest-Collection_TM_QMUL_28_6_23'
file_extension = '.scn'

import os
import os
def count_files_with_extension(directory, extension):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


# Call the function to count the files with the specified extension
file_count = count_files_with_extension(directory_path, file_extension)

print(f"Total number of files with extension '{file_extension}': {file_count}")


def count_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

# Call the function to count the files
file_count = count_files(directory_path)

file_path = os.path.join(directory_path, 'NH13-581.scn')

file_path = r'D:/_Guest-Collection_TM_QMUL_28_6_23/Batch 001/NH13-581/NH13-581.scn'

# following https://www.youtube.com/watch?v=QntLBvUZR5c tutorial
from openslide import open_slide
import openslide 
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

slide = open_slide(file_path)

slide_props = slide.properties
print(slide_props)

print("Vendor is:", slide_props['openslide.vendor'])
print("Pixel size of X in um is:", slide_props['openslide.mpp-x'])
print("Pixel size of Y in um is:", slide_props['openslide.mpp-y'])

#Objective used to capture the image
objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
print("The objective power is: ", objective)


# get slide dimensions for the level 0 - max resolution level
slide_dims = slide.dimensions
print(slide_dims)


#Get a thumbnail of the image and visualize
slide_thumb_600 = slide.get_thumbnail(size=(600, 1000))
slide_thumb_600.show()

#Convert thumbnail to numpy array
slide_thumb_600_np = np.array(slide_thumb_600)
plt.figure(figsize=(8,8))
plt.imshow(slide_thumb_600_np)    


dims = slide.level_dimensions

#By how much are levels downsampled from the original image?
factors = slide.level_downsamples
print("Each level is downsampled by an amount of: ", factors)


#Copy an image from a level
level3_dim = dims[2]
#Give pixel coordinates (top left pixel in the original large image)
#Also give the level number (for level 3 we are providing a valueof 2)
#Size of your output image
#Remember that the output would be a RGBA image (Not, RGB)
level3_img = slide.read_region((0,0), 2, level3_dim) #Pillow object, mode=RGBA

#Convert the image to RGB
level3_img_RGB = level3_img.convert('RGB')
level3_img_RGB.show()

#Convert the image into numpy array for processing
level3_img_np = np.array(level3_img_RGB)
plt.imshow(level3_img_np)

#Return the best level for displaying the given downsample.
SCALE_FACTOR = 32
best_level = slide.get_best_level_for_downsample(SCALE_FACTOR)

