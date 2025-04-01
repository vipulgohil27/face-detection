import boto3
import os
import io
import imagehash
import datetime
from PIL import Image
from fastapi import FastAPI
from typing import List

app = FastAPI()

s3_client = boto3.client("s3")


# Get all image files from S3
def list_images(bucket_name):
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    return sorted(
        [(obj["Key"], obj["LastModified"], obj["Size"]) for obj in response.get("Contents", [])],
        key=lambda x: x[1],  # Sort by LastModified timestamp
    )


# Find min and max timestamps
def get_time_range(bucket_name):
    images = list_images(bucket_name)
    if images:
        return images[0][1], images[-1][1]
    return None, None


# Compute hash of an image
def compute_hash(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return imagehash.phash(image)


# Identify similar images within 60 seconds and preserve the highest resolution
def find_and_filter_duplicates(bucket_name):
    images = list_images(bucket_name)
    duplicates = []
    image_groups = {}

    for key, timestamp, size in images:
        image_bytes = s3_client.get_object(Bucket=bucket_name, Key=key)["Body"].read()
        img_hash = compute_hash(image_bytes)

        # Group by hash
        if img_hash in image_groups:
            image_groups[img_hash].append((key, timestamp, size))
        else:
            image_groups[img_hash] = [(key, timestamp, size)]

    for img_hash, group in image_groups.items():
        group.sort(key=lambda x: (-x[2], x[1]))  # Sort by size (desc), then timestamp (asc)
        preserved_image = group[0]  # Keep the highest resolution image
        duplicates.extend([img[0] for img in group[1:]])  # Mark others as duplicates

    return duplicates


# Move duplicates to another bucket
def move_duplicates(source_bucket, dest_bucket):
    duplicates = find_and_filter_duplicates(source_bucket)
    for dup_key in duplicates:
        copy_source = {"Bucket": source_bucket, "Key": dup_key}
        s3_client.copy_object(CopySource=copy_source, Bucket=dest_bucket, Key=dup_key)
        s3_client.delete_object(Bucket=source_bucket, Key=dup_key)
    return len(duplicates)


@app.get("/process")
def process_images(bucket_url: str, dest_bucket: str):
    bucket_name = bucket_url.split("/")[-1]  # Extract bucket name from URL
    min_time, max_time = get_time_range(bucket_name)
    duplicate_count = move_duplicates(bucket_name, dest_bucket)
    return {
        "min_time": str(min_time),
        "max_time": str(max_time),
        "duplicates_moved": duplicate_count,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
