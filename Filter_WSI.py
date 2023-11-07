# Filter Image Step ONE: Moving images from nested folders
import os
import shutil
from openslide import open_slide
from PIL import Image
import matplotlib.pyplot as plt

# Batch number
batch_number = "080"

# Path to the source directory
source_directory = r"D:\Guest-Collection_TM_QMUL_28_6_23\Batch {}".format(batch_number)

# Get the base folder name from the source directory
base_folder_name = os.path.basename(source_directory)

# Destination directory where you want to move the images
destination_directory = os.path.join(r"D:\Phd\DNA_meth\dataset\img", base_folder_name)

# Counter for moved images
moved_image_count = 0

# Create the destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Iterate through the source directory and its subdirectories
for foldername, subfolders, filenames in os.walk(source_directory):
    for filename in filenames:
        # Check if the file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.scn', '.ndpi')):
            # Get the full path of the source file
            source_file = os.path.join(foldername, filename)
            # Get the destination file path by joining the destination directory and the filename
            destination_file = os.path.join(destination_directory, filename)
            # Move the file from the source to the destination folder
            shutil.move(source_file, destination_file)
            # Increment the moved image counter
            moved_image_count += 1

print(f"Moved {moved_image_count} images to '{destination_directory}' successfully.")

#       ###################################################################
# Step TWO: Ploting imgs to know the stain type

# Path to the folder containing SCN images
lis = os.listdir(r"D:\Phd\DNA_meth\dataset\img\Batch {}".format(batch_number))
input_folder = r"D:\Phd\DNA_meth\dataset\img\Batch {}/".format(batch_number)


i=0
# Loop through the SCN files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.scn', '.ndpi')):  # images are in SCN format
        # Open the SCN file using openslide
        scn_path = os.path.join(input_folder, filename)
        slide = open_slide(scn_path)
        
        # Get the image thumbnail
        slide_thumb_600 = slide.get_thumbnail(size=(512, 512))
        
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
