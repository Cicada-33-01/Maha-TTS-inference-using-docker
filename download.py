import os
import requests
import zipfile
import torch
import glob
from maha_tts import load_models, infer_tts, config
from scipy.io.wavfile import write

# Define the folder name
folder_name = "generated-audio"

# Create the folder if it doesn't already exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")

# Function to download a file
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        f.write(response.content)


# Download and unzip the reference speaker wav files
url = "https://huggingface.co/Dubverse/MahaTTS/resolve/main/maha_tts/pretrained_models/infer_ref_wavs.zip"
zip_file = "infer_ref_wavs.zip"
extract_path = "infer_ref_wavs"

# Download the file
download_file(url, zip_file)

# Unzip the file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Define the speaker paths
speakers = [
    os.path.join(extract_path, '2272_152282_000019_000001/'),
    os.path.join(extract_path, '2971_4275_000049_000000/'),
    os.path.join(extract_path, '4807_26852_000062_000000/'),
    os.path.join(extract_path, '6518_66470_000014_000002/')
]
