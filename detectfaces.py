import os
import cv2
import pandas as pd

# Load pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def extract_faces(image_path, output_dir):
    """Detects faces in an image and saves them as separate files."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_count = len(faces)

    # Save extracted faces
    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y + h, x:x + w]
        face_filename = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_face_{i}.jpg")
        cv2.imwrite(face_filename, face)

    return face_count


def process_images(input_dir, output_csv, output_faces_dir):
    """Processes images in a directory, extracts faces, and writes results to CSV."""
    os.makedirs(output_faces_dir, exist_ok=True)
    data = []
    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(input_dir, image_name)
            count = extract_faces(image_path, output_faces_dir)
            data.append([image_name, count])

    # Save results to CSV
    df = pd.DataFrame(data, columns=['Image Name', 'Face Count'])
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# Example usage
input_directory = "./images"  # Directory containing images
output_file = "face_counts.csv"
output_faces_directory = "./faces"  # Directory to save extracted faces
process_images(input_directory, output_file, output_faces_directory)
