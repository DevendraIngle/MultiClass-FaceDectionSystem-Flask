# MultiClass-FaceDectionSystem-Flask
![Landing Page](https://github.com/DevendraIngle/MultiClass-FaceDectionSystem-Flask/blob/main/Landing%20page.png?raw=true)
## :bulb: Objective :

Develop a face recognition system that can take user input images and live video, preprocess them, store them, and recognize faces using DeepFace and the VGG16 model, displaying results through a web interface and is to develop a robust and user-friendly face recognition system that leverages advanced deep learning techniques to enhance security and personalization in various applications.

## Project Outline :

1. **Image Recognition with Haarcascade and OpenCV**
2. **Image Data Preprocessing**
3. **Face Recognition Classification Model with VGG16**
4. **Flask (HTML, CSS, HTTP Methods)**

Finally, we will integrate all these components to build a fully functional face recognition web app.

## 🚀&nbsp;Getting Started

### 1. Clone the Repository

```bash
https://github.com/DevendraIngle/MultiClass-FaceDectionSystem-Flask.git
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

- On Windows:
```bash
venv\Scripts\activate
```
- On macOS and Linux:
```bash
source venv/bin/activate
```
### 4. Install Dependencies
```bash
pip install -r requirements.txt
```
Restart venv to avoid any issues.
### 5. Run the App
```bash
python main.py
```
Open your web browser and go to http://127.0.0.1:5000/ to see the app in action.

### 6. Usage

1. **Home Page**: 
   - Open the application.
   - Click on the "Get Started Today" button to proceed.

2. **Login Page**:
   - Enter the credentials: 
     - **Username**: `admin`
     - **Password**: `admin`
   - Click "Login" to access the Navigation Page.

3. **Navigation Page**:
   - You'll see three options:
     1. **Upload Images for Training**: Upload images for face recognition.
     2. **Train Data**: Train the model using the uploaded images.
     3. **Face Detection**: Use the trained model to predict faces in new images.

4. **Upload Images for Training**:
   - Click on the  option.
   - Upload your images with the name.
   - The images will be preprocessed and stored automatically in the backend.

5. **Train the Model**:
   - Click on the "Train" option.
   - The model will train using the preprocessed images and save embeddings

6. **Predict Faces**:
   - Click on the "Face Detection" option.
   - Upload a new image to predict and recognize faces using the trained model.
   - The prediction results will be displayed on the screen you can choose between live-video-feed or on static images.

# :cyclone: System Design

## Modules

### 1. User Interface (UI) Module
- **Description**: 
  - Develop HTML/CSS-based web pages for user interaction.
  - Implement forms for uploading images and accessing the live video feed.

### 2. Image/Video Capture Module
- **Description**: 
  - Capture images or live video from the user's device.
  - Provide functionality for users to upload images manually.

### 3. Preprocessing Module
- **Description**: 
  - Convert images to grayscale, resize, and normalize.
  - Detect and align faces within the images/video frames using OpenCV or similar libraries.

### 4. Face Recognition Module
- **Description**: 
  - Use DeepFace and VGG16 for feature extraction and face recognition.
  - Implement logic to compare input images with stored images for recognition.

### 5. Database Module
- **Description**: 
  - Store preprocessed images, extracted features in respected upload and predict folders

### 6. Results Display Module
- **Description**: 
  - Display recognized faces along with user details on the web interface.
  - Show real-time recognition results on the live video feed.

## System Flow ([Figma](https://www.figma.com/design/PYza59lyc3BrfsGVIkuRkx/MultiClass-Face-Detection-System-(View)?m=auto&t=iAkc1cWma8sntGfT-1))
1. **User Access**:
   - Users can upload images or initiate live video capture through the web interface.
  
2. **Preprocessing**:
   - Uploaded images or video frames are preprocessed to ensure consistency in face detection and recognition.

3. **Face Recognition**:
   - The preprocessed images are passed through the face recognition module, where features are extracted and compared against stored data.

4. **Results Storage**:
   - Recognition results, along with preprocessed images and metadata, are stored in dediacted folders.

5. **Results Display**:
   - The results of the recognition process are displayed on the web interface, updating in real-time for live video feeds.

# [Deepface](https://github.com/serengil/deepface)
<p align="center"><img src="https://github.com/user-attachments/assets/96014b9d-89c1-4eee-b8b7-f99e14f7e17c" width="95%" height="95%"></p>

**Face recognition models** 

DeepFace is a **hybrid** face recognition package. It currently wraps many **state-of-the-art** face recognition models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) , [`FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/), [`ArcFace`](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/), [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/), `SFace` and `GhostFaceNet`. The default configuration uses VGG-Face model.

```python
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

#face verification
result = DeepFace.verify(
  img1_path = "img1.jpg",
  img2_path = "img2.jpg",
  model_name = models[0],
)

#face recognition
dfs = DeepFace.find(
  img_path = "img1.jpg",
  db_path = "C:/workspace/my_db", 
  model_name = models[1],
)

#embeddings
embedding_objs = DeepFace.represent(
  img_path = "img.jpg",
  model_name = models[2],
)
```

FaceNet, VGG-Face, ArcFace and Dlib are overperforming ones based on experiments - see [`BENCHMARKS`](https://github.com/serengil/deepface/tree/master/benchmarks) for more details. You can find the measured scores of various models in DeepFace and the reported scores from their original studies in the following table.

| Model          | Measured Score | Declared Score     |
| -------------- | -------------- | ------------------ |
| Facenet512     | 98.4%          | 99.6%              |
| Human-beings   | 97.5%          | 97.5%              |
| Facenet        | 97.4%          | 99.2%              |
| Dlib           | 96.8%          | 99.3 %             |
| VGG-Face       | 96.7%          | 98.9%              |
| ArcFace        | 96.7%          | 99.5%              |
| GhostFaceNet   | 93.3%          | 99.7%              |
| SFace          | 93.0%          | 99.5%              |
| OpenFace       | 78.7%          | 92.9%              |
| DeepFace       | 69.0%          | 97.3%              |
| DeepID         | 66.5%          | 97.4%              |

Conducting experiments with those models within DeepFace may reveal disparities compared to the original studies, owing to the adoption of distinct detection or normalization techniques. Furthermore, some models have been released solely with their backbones, lacking pre-trained weights. Thus, we are utilizing their re-implementations instead of the original pre-trained weights.

For more information and on how to use the library go to library `https://github.com/serengil/deepface`

# VGG-16

A convolutional neural network is also known as a ConvNet, which is a kind of artificial neural network. A convolutional neural network has an input layer, an output layer, and various hidden layers. VGG16 is a type of CNN (Convolutional Neural Network) that is considered to be one of the best computer vision models to date. The creators of this model evaluated the networks and increased the depth using an architecture with very small (3 × 3) convolution filters, which showed a significant improvement on the prior-art configurations. They pushed the depth to 16–19 weight layers making it approx — 138 trainable parameters.

<p align="center"><img src="https://github.com/user-attachments/assets/66c3615b-7398-455e-88b8-400d1474820b" width="95%" height="95%"></p>
<p align="center"><img src="https://github.com/user-attachments/assets/60511a4d-5bd1-45ef-bcc7-1ac74a0c0ac5" width="95%" height="95%"></p>

### Features
- The 16 in VGG16 refers to 16 layers that have weights. In VGG16 there are thirteen convolutional layers, five Max Pooling layers, and three Dense layers which sum up to 21 layers but it has only sixteen weight layers i.e., learnable parameters layer.
VGG16 takes input tensor size as 224, 244 with 3 RGB channel
- Most unique thing about VGG16 is that instead of having a large number of hyper-parameters they focused on having convolution layers of 3x3 filter with stride 1 and always used the same padding and maxpool layer of 2x2 filter of stride 2.
- The convolution and max pool layers are consistently arranged throughout the whole architecture
- Conv-1 Layer has 64 number of filters, Conv-2 has 128 filters, Conv-3 has 256 filters, Conv 4 and Conv 5 has 512 filters.
- Three Fully-Connected (FC) layers follow a stack of convolutional layers: the first two have 4096 channels each, the third performs 1000-way ILSVRC classification and thus contains 1000 channels (one for each class). The final layer is the soft-max layer.

### Architecture
- **Inputs** : The VGGNet accepts 224224-pixel images as input. To maintain a consistent input size for the ImageNet competition, the model’s developers chopped out the central 224224 patches in each image.
- **Convolutional Layers** : VGG’s convolutional layers use the smallest feasible receptive field, or 33, to record left-to-right and up-to-down movement. Additionally, 11 convolution filters are used to transform the input linearly. The next component is a ReLU unit, a significant advancement from AlexNet that shortens training time. Rectified linear unit activation function, or ReLU, is a piecewise linear function that, if the input is positive, outputs the input; otherwise, the output is zero. The convolution stride is fixed at 1 pixel to keep the spatial resolution preserved after convolution (stride is the number of pixel shifts over the input matrix).
- **Hidden Layers** : The VGG network’s hidden layers all make use of ReLU. Local Response Normalization (LRN) is typically not used with VGG as it increases memory usage and training time. Furthermore, it doesn’t increase overall accuracy.
- **Fully Connected Layers** : The VGGNet contains three layers with full connectivity. The first two levels each have 4096 channels, while the third layer has 1000 channels with one channel for each class.

<p align="center"><img src="https://github.com/user-attachments/assets/9957ebd7-ba4a-46f4-a4ab-b34e3da62765" width="95%" height="95%"></p>

### VGG-16-Summary

<p align="center"><img src="https://github.com/user-attachments/assets/ff715117-0f66-4492-bc4c-65fb8783874b" width="95%" height="95%"></p>

## Guidance for Manual Training

### VGG-16 Implementation

```python 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

_input = Input((224,224,1)) 

conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
pool1  = MaxPooling2D((2, 2))(conv2)

conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
pool2  = MaxPooling2D((2, 2))(conv4)

conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
pool3  = MaxPooling2D((2, 2))(conv7)

conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
pool4  = MaxPooling2D((2, 2))(conv10)

conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
pool5  = MaxPooling2D((2, 2))(conv13)

flat   = Flatten()(pool5)
dense1 = Dense(4096, activation="relu")(flat)
dense2 = Dense(4096, activation="relu")(dense1)
output = Dense(1000, activation="softmax")(dense2)

vgg16_model  = Model(inputs=_input, outputs=output)
```
### Working with pretrained model

```python 
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt 
from PIL import Image 
import seaborn as sns
import pandas as pd 
import numpy as np 
import os 

img1 = "../input/flowers-recognition/flowers/tulip/10094729603_eeca3f2cb6.jpg"
img2 = "../input/flowers-recognition/flowers/dandelion/10477378514_9ffbcec4cf_m.jpg"
img3 = "../input/flowers-recognition/flowers/sunflower/10386540696_0a95ee53a8_n.jpg"
img4 = "../input/flowers-recognition/flowers/rose/10090824183_d02c613f10_m.jpg"
imgs = [img1, img2, img3, img4]

def _load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img 

def _get_predictions(_model):
    f, ax = plt.subplots(1, 4)
    f.set_size_inches(80, 40)
    for i in range(4):
        ax[i].imshow(Image.open(imgs[i]).resize((200, 200), Image.ANTIALIAS))
    plt.show()
    
    f, axes = plt.subplots(1, 4)
    f.set_size_inches(80, 20)
    for i,img_path in enumerate(imgs):
        img = _load_image(img_path)
        preds  = decode_predictions(_model.predict(img), top=3)[0]
        b = sns.barplot(y=[c[1] for c in preds], x=[c[2] for c in preds], color="gray", ax=axes[i])
        b.tick_params(labelsize=55)
        f.tight_layout()
```

### Utilizing Pretrained Weights if Training Becomes Time-Consuming

```python
from keras.applications.vgg16 import VGG16
vgg16_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
vgg16_model = VGG16(weights=vgg16_weights)
_get_predictions(vgg16_model)

'''
Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
40960/35363 [==================================] - 0s 0us/step
'''
```
