import time
import face_recognition
import cv2
import numpy as np
import os
import pickle

class FaceRecognitionModel:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def rotate_image(self, image, angle):
        """Rotate the image by the given angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, matrix, (w, h))
        return rotated_image

    def train_model(self, images_path):
        """
        Trains the model with images stored in the given directory.
        Each image should be named as '<name>.jpg'.
        """
        for filename in os.listdir(images_path):
            if filename.endswith('.jpg'):
                image_path = os.path.join(images_path, filename)
                try:
                    # Load the original image
                    image = face_recognition.load_image_file(image_path)
                    found_face = False
                    for angle in range(0, 360, 90):  # Rotate by 90 degrees increments
                        # Rotate the image
                        rotated_image = self.rotate_image(image, angle)
                        # Find face encodings
                        face_encodings = face_recognition.face_encodings(rotated_image)
                        if face_encodings:  # If at least one face is found
                            self.known_face_encodings.append(face_encodings[0])
                            self.known_face_names.append(os.path.splitext(filename)[0])  # Use filename without extension
                            found_face = True
                            print(f"Found face in {filename} at {angle} degrees.")
                            break  # Exit the loop if a face is found

                    if not found_face:
                        print(f"No faces found in {filename} after rotations.")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    def save_model(self, file_path):
        """Saves the known face encodings and names to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump((self.known_face_encodings, self.known_face_names), f)

    def load_model(self, file_path):
        """Loads the known face encodings and names from a file."""
        with open(file_path, 'rb') as f:
            self.known_face_encodings, self.known_face_names = pickle.load(f)

    def recognize_faces(self, image_path):
        """Recognizes faces from a single image, rotating until a face is found."""
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to load image.")
            return

        # Convert the image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for angle in range(0, 360, 90):
            rotated_image = self.rotate_image(rgb_image, angle)
            face_encodings = face_recognition.face_encodings(rotated_image)
            if face_encodings:  # Check if any face encodings were found
                # Compare the found face encoding with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[0])
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encodings[0])

                # Find the index of the smallest distance
                if matches:  # Only proceed if there are any matches
                    # Get the index of the smallest distance among matches
                    best_match_index = np.argmin(face_distances)
                    best_match_distance = face_distances[best_match_index]

                    # Check if the best match is indeed a match
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        print(f'The image is for: {name} with a distance of {best_match_distance:.2f}')
                        return  # Exit after finding the best match

        print("No faces found in the image after rotating.")

# Example usage:
if __name__ == "__main__":
    model = FaceRecognitionModel()

    # Train the model with images
    model.train_model('./dataset1/')  # Directory containing face images

    # Save the model to a file
    model.save_model('face_model.pkl')

    # Load the model (this could be done in a separate instance or after training)
    model.load_model('face_model.pkl')

    # Start recognizing faces using a specific image
    model.recognize_faces('1.jpg')  # Replace with your image path