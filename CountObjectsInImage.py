import os
import cv2
import torch
import pandas as pd
from pathlib import Path
from torchvision import transforms

# Load pre-trained YOLOv5 model for fruit detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def count_fruits(image_path):
    """Detects fruits in an image and returns the count."""
    img = cv2.imread(image_path)
    results = model(img)
    detections = results.pandas().xyxy[0]  # Get detection results as Pandas DataFrame
    fruit_count = len(detections)  # Count total detected objects (modify if filtering for fruits only)
    return fruit_count


def process_images(input_dir, output_csv):
    """Processes images in a directory, counts fruits, and writes results to CSV."""
    data = []
    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(input_dir, image_name)
            count = count_fruits(image_path)
            data.append([image_name, count])

    # Save results to CSV
    df = pd.DataFrame(data, columns=['Image Name', 'Fruit Count'])
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# Example usage
input_directory = "./images"  # Directory containing images
output_file = "fruit_counts.csv"
process_images(input_directory, output_file)
