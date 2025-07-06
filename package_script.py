import zipfile
import os

# Define the folder path and output ZIP file
folder_path = '.'
zip_filename = 'SmartEraseVR_app.zip'

# Get a list of all files in the folder
files = []
for root, dirs, filenames in os.walk(folder_path):
    for filename in filenames:
        files.append(os.path.join(root, filename))

# Create a new ZIP file
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for file in files:
        if file == zip_filename:  # Avoid adding the ZIP file itself
            continue
        zipf.write(file, arcname=os.path.relpath(file, folder_path))

print(f"Project files zipped into '{zip_filename}'")