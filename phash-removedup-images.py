import imagehash
import os
import io
import imagehash
import datetime
from PIL import Image
from fastapi import FastAPI
from typing import List


def compute_phash(img_path):
    img = Image.open(img_path).convert("RGB")
    return imagehash.phash(img)

# Get all image files from a local folder
def list_images(folder_path):
    images = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            timestamp = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
            size = os.path.getsize(file_path)
            images.append((file_path, timestamp, size))
    return images

def find_similar_images(folder_path):
    images = list_images(folder_path)
    similar_images = []

    hashes = {img[0]: compute_phash(img[0]) for img in images}

    for img1, hash1 in hashes.items():
        for img2, hash2 in hashes.items():
            if img1 != img2 and hash1 - hash2 < 5:  # Threshold for similarity
                similar_images.append((img1, img2, hash1 - hash2))

    return similar_images
