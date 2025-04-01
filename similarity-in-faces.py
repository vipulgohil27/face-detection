import face_recognition
import cv2
import os
import numpy as np


# Function to load images and detect face encodings
def load_face_encodings(image_folder):
    face_encodings = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            # We assume there's one face per image, you can modify this to handle multiple faces
            if len(face_locations) > 0:
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                face_encodings[filename] = encoding

    return face_encodings


# Function to compare all faces and find similar ones
def find_similar_faces(face_encodings, threshold=0.6):
    similar_faces = []

    filenames = list(face_encodings.keys())

    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            file1 = filenames[i]
            file2 = filenames[j]

            # Compare faces
            encoding1 = face_encodings[file1]
            encoding2 = face_encodings[file2]

            # Compute similarity
            distance = np.linalg.norm(encoding1 - encoding2)

            if distance < threshold:
                similar_faces.append((file1, file2, distance))

    return similar_faces


# Main execution
def main(image_folder):
    # Load face encodings from images in folder
    face_encodings = load_face_encodings(image_folder)

    if len(face_encodings) == 0:
        print("No faces found in the folder.")
        return

    # Find similar faces
    similar_faces = find_similar_faces(face_encodings)

    if similar_faces:
        print("Similar faces found:")
        for face1, face2, dist in similar_faces:
            print(f"{face1} and {face2} are similar with a distance of {dist}")
    else:
        print("No similar faces found.")


# Example usage
image_folder = "C:\\Users\\vipul\\PycharmProjects\\PythonProject\\faces"
main(image_folder)
