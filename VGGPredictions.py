import os
import cv2
import pandas as pd
from deepface import DeepFace
import numpy as np
import shutil

# Load the Haar Cascade classifier from the specified file path
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

# Load embeddings from CSV file
def load_embeddings(csv_path):
    return pd.read_csv(csv_path)

# Function to compute cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_a = np.linalg.norm(vector1)
    norm_b = np.linalg.norm(vector2)
    return dot_product / (norm_a * norm_b)

# Function to find the best match for an embedding from the saved embeddings
def find_best_match(embedding, embeddings_df, threshold=0.6):
    best_match = "Unknown"
    highest_similarity = -1
    
    for idx, row in embeddings_df.iterrows():
        saved_embedding = eval(row['embedding'])  # Convert string representation back to a list
        similarity = cosine_similarity(embedding, saved_embedding)
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = row['identity']
    
    # Check if the highest similarity is above the threshold
    if highest_similarity < threshold:
        best_match = "Unknown"
    
    return best_match, highest_similarity

# Function to detect faces in an image
def detect_faces(image_path):
    img = cv2.imread(image_path)  # Read image in BGR
    if img is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print(f"No faces detected in the image {image_path}")
    
    return img, faces

# Function to draw bounding boxes based on names instead of predictions
def predict_image(img, faces, name):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Calculate text size and position
        text_width, text_height = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = x + (w - text_width) // 2
        text_y = y - 10
        
        # Draw filled rectangle for text background
        cv2.rectangle(img, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 255, 0), cv2.FILLED)
        
        # Draw text
        cv2.putText(img, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
    return img

# Function to generate embedding for a new image
def get_embedding(image_path):
    embedding = DeepFace.represent(img_path=image_path, model_name='VGG-Face', enforce_detection=False)
    return embedding[0]['embedding'] if embedding else None

def PredictImage():

    # Load the saved embeddings
    embeddings_df = load_embeddings('models/embeddings.csv')
    
    # Clear the result directory
    result_dir = 'predict/result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    # Test the function with an image from predict/upload
    upload_dir = 'predict/upload'
    test_image_path = os.path.join(upload_dir, os.listdir(upload_dir)[0])  # Assuming there is one image in the upload folder

    # Detect faces in the test image
    test_img, faces = detect_faces(test_image_path)

    # Generate the embedding for the new image
    new_image_embedding = get_embedding(test_image_path)

    if new_image_embedding:
        # Find the best match from the saved embeddings with a threshold
        best_match, similarity = find_best_match(new_image_embedding, embeddings_df, threshold=0.6)
        name = best_match if best_match else "Unknown"
    else:
        name = "Unknown"

    # Draw bounding boxes based on the extracted name
    test_img = predict_image(test_img, faces, name)

    # Save the result image
    result_image_path = os.path.join(result_dir, 'result.jpg')
    cv2.imwrite(result_image_path, test_img)

    print(f"Result saved at {result_image_path}")

def GenerateFrames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video capture.")

    embeddings_df = load_embeddings('models/embeddings.csv')  # Load the saved embeddings

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                embedding = get_embedding(face_img)
                if embedding:
                    best_match, similarity = find_best_match(embedding, embeddings_df, threshold=0.6)
                    name = best_match if best_match else "Unknown"
                    frame = predict_image(frame, [(x, y, w, h)], name)
                else:
                    frame = predict_image(frame, [(x, y, w, h)], "Unknown")

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()