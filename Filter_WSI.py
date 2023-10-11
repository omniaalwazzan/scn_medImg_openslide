# This involves manual filtering as folders have mixed types of HCI and WSI, but this scrip helps in automize part of the manual selection 
import os
from openslide import open_slide
from PIL import Image
import matplotlib.pyplot as plt
# Path to the folder containing SCN images
lis = os.listdir(r'D:\Guest-Collection_TM_QMUL_28_6_23\Batch 015')
#lis = ['NH16-580', 'NH16-623', 'NH16-862']
input_folder = r"D:\Guest-Collection_TM_QMUL_28_6_23\Batch 015" + '/' + lis[2] 
#\NH13-1014

i=0
#input_folder =r"D:\Guest-Collection_TM_QMUL_28_6_23\Batch 005\NH08-210"
#C:\Users\Omnia\Desktop\Batch 001\NH13-1256
# Loop through the SCN files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.scn'):  # images are in SCN format
        # Open the SCN file using openslide
        scn_path = os.path.join(input_folder, filename)
        slide = open_slide(scn_path)
        
        # Get the image thumbnail
        slide_thumb_600 = slide.get_thumbnail(size=(1000, 3000))
        
        # Plot the thumbnail
        plt.imshow(slide_thumb_600)
        
        # Plot the image name as a title
        plt.title(filename)
        
        # Display the plot
        plt.show()
        
        print(f"this is image {i+1}")

        print("\n --------------------\n")
        i+=1


##################################


