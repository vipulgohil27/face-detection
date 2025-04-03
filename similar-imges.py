import os
import io
import datetime
import numpy as np
from PIL import Image
from fastapi import FastAPI
from typing import List, Tuple
from skimage.metrics import structural_similarity as ssim

app = FastAPI()


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


# Find min and max timestamps
def get_time_range(folder_path):
    images = list_images(folder_path)
    if images:
        return images[0][1], images[-1][1]
    return None, None


# Compute SSIM similarity between two images
# def compute_similarity(img1_path, img2_path):
#     img1 = Image.open(img1_path).convert("L").resize((256, 256))
#     img2 = Image.open(img2_path).convert("L").resize((256, 256))
#     img1_np = np.array(img1)
#     img2_np = np.array(img2)
#     return ssim(img1_np, img2_np)

def compute_similarity(img1_path, img2_path):
    img1 = Image.open(img1_path).convert("RGB").resize((256, 256))
    img2 = Image.open(img2_path).convert("RGB").resize((256, 256))
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    return ssim(img1_np, img2_np, multichannel=True)


# Identify similar images with similarity > 90%
def find_similar_images(folder_path):
    images = list_images(folder_path)
    similar_images = []

    for i in range(len(images)):
        #print("i=%s",i)
        #for j in range(i + 1, len(images)):
        if i < len(images) -10:
            count=i
        else:
            count=len(images)
        for j in range(i + 1, count):
            similarity = compute_similarity(images[i][0], images[j][0])
            print(similarity,images[i][0], images[j][0])
            if similarity > 0.60:
                similar_images.append((images[i][0], images[j][0], similarity))
                print(similar_images)

    return similar_images


@app.get("/process")
def process_images(folder_path: str):
    folder_path="C:/Users/vipul/Downloads/sample"
    min_time, max_time = get_time_range(folder_path)
    similar_images = find_similar_images(folder_path)
    print("done")
    return {
        "min_time": str(min_time),
        "max_time": str(max_time),
        "similar_images": similar_images,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
