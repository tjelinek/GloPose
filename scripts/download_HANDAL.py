import os
import gdown
import zipfile

def download_and_unzip_gdrive_folder(folder_url, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download the folder's contents using gdown
    gdown.download_folder(folder_url, output=output_dir, quiet=False)

    # Unzip all zip files in the output directory
    for item in os.listdir(output_dir):
        if item.endswith('.zip'):
            file_path = os.path.join(output_dir, item)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(file_path)  # Remove the zip file after extracting

# Example usage
folder_url = 'https://drive.google.com/drive/folders/1B5r7CO5gEoqFl_K4PAikg7pYYflbyuo_'  # Replace with your folder URL
output_dir = '/mnt/personal/jelint19/data/HANDAL'

download_and_unzip_gdrive_folder(folder_url, output_dir)
