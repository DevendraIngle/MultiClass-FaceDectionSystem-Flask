import os
import cv2
import numpy as np
from glob import glob

def preprocess_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image_paths = glob(os.path.join(input_folder, '*'))
    haar = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    
    for image_path in image_paths:
        try:
            # Step -1: Read Image and Convert to RGB
            img = cv2.imread(image_path)  # read image in BGR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image from BGR to RGB

            # Step -2: Apply Haar Cascade Classifier
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)  # convert image to grayscale
            faces_list = haar.detectMultiScale(gray, 1.5, 5)

            for x, y, w, h in faces_list:
                # Step -3: Crop Face
                roi = img[y:y + h, x:x + w]

                # Step -4: Save Image
                image_name = os.path.basename(image_path)
                output_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_path, roi)
                print(f'Image {image_name} successfully processed')

        except Exception as e:
            print(f'Unable to process the image {image_path}: {str(e)}')

