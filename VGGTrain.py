import os
from deepface import DeepFace
import pandas as pd
from tqdm import tqdm

def TrainModel():
    database_path = 'uploads/Cropped_Data'
    output_csv_path = 'models/embeddings.csv'
    
    def create_embeddings(database_path):
        embeddings = []
        people = os.listdir(database_path)
        for person_name in tqdm(people, desc="Processing people"):
            person_dir = os.path.join(database_path, person_name)
            if os.path.isdir(person_dir):
                images = os.listdir(person_dir)
                for image_name in tqdm(images, desc=f"Processing images in {person_name}", leave=False):
                    image_path = os.path.join(person_dir, image_name)
                    if image_path.lower().endswith(('png', 'jpg', 'jpeg')):
                        embedding = DeepFace.represent(img_path=image_path, model_name='VGG-Face', enforce_detection=False)
                        for face in embedding:
                            face['identity'] = person_name
                            embeddings.append(face)
        return pd.DataFrame(embeddings)
    
    embeddings_df = create_embeddings(database_path)
    embeddings_df.to_csv(output_csv_path, index=False)
