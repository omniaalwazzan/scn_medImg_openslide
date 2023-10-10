import os
from openslide import open_slide
from PIL import Image
import matplotlib.pyplot as plt
# Path to the folder containing SCN images
# C:\Users\Omnia\Desktop\Batch 001\NH14-2331
input_folder = r'C:\Users\Omnia\Desktop\Batch 001\NH15-40'
#C:\Users\Omnia\Desktop\Batch 001\NH13-1256
# Loop through the SCN files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.scn'):  # images are in SCN format
        # Open the SCN file using openslide
        scn_path = os.path.join(input_folder, filename)
        slide = open_slide(scn_path)
        
        # Get the image thumbnail
        slide_thumb_600 = slide.get_thumbnail(size=(3000, 3000))
        
        # Plot the thumbnail
        plt.imshow(slide_thumb_600)
        
        # Plot the image name as a title
        plt.title(filename)
        
        # Display the plot
        plt.show()
        
        print("\n--------------------\n")

