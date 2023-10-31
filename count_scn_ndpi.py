import os

# Directory containing your top-level folders
top_level_directory = r"D:\Guest-Collection_TM_QMUL_28_6_23"

# Dictionary to store counts for different extensions
extension_counts = {
    ".scn": 0,
    ".ndpi": 0
}

# Function to count files with specific extensions in nested folders
def count_files_with_extension(directory, extensions, counts):
    for root, dirs, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in extensions:
                counts[ext] += 1

# List of extensions you are interested in
extensions_to_count = [".scn", ".ndpi"]

# Count the number of specified extensions in nested folders
count_files_with_extension(top_level_directory, extensions_to_count, extension_counts)

# Print the counts
for ext, count in extension_counts.items():
    print(f"Number of {ext} files: {count}")

file_name = []
for root, dirs, files in os.walk(top_level_directory):
    for file in files:
        _, ext = os.path.splitext(file)
        print(file)
        file_name.append(file)
