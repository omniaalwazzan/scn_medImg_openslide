import os
import csv

root_directory = r"D:\_Guest-Collection_TM_QMUL_28_6_23"
output_csv = r"E:\2ndYear\Slivia's project\Silvia Dataset\output.csv"


# Open the CSV file in write mode
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Iterate over the folders in the root directory
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        
        # Check if the item is a directory and starts with "Batch"
        if os.path.isdir(folder_path) and folder_name.startswith("Batch"):
            batch_number = folder_name.split(" ")[-1]  # Extract the batch number
            
            # Iterate over the subfolders in the batch folder
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                
                # Check if the subfolder starts with "NH" and is a directory
                if os.path.isdir(subfolder_path) and subfolder_name.startswith("NH"):
                    writer.writerow([batch_number, subfolder_name])

print("CSV file created successfully.")
