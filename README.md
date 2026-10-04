# 🖼️ CIFAR-10 CNN Image Classifier

A **Deep Learning image classification web application** built using **PyTorch** and **Streamlit**.

This project uses a **Convolutional Neural Network (CNN)** trained on the **CIFAR-10 dataset** to classify images into 10 different categories.

The trained model is integrated into a Streamlit web application where users can upload an image and get the predicted category along with the model's confidence.

---

## 🚀 Live Demo

🔗 **Live App:** *https://cnn-image-classifierr.streamlit.app/*

---

## 📌 Project Overview

The goal of this project is to build an end-to-end image classification system using a CNN.

The complete workflow is:

```text
CIFAR-10 Dataset
       ↓
Image Preprocessing
       ↓
CNN Model
       ↓
Model Training
       ↓
Validation
       ↓
Best Model Saved
       ↓
Streamlit Web Application
       ↓
Image Upload
       ↓
Prediction + Confidence
```

---

## 🗂️ CIFAR-10 Dataset

The model is trained on the **CIFAR-10 dataset**, which contains images belonging to 10 classes.

### Supported Categories

* ✈️ Airplane
* 🚗 Automobile
* 🐦 Bird
* 🐱 Cat
* 🦌 Deer
* 🐶 Dog
* 🐸 Frog
* 🐴 Horse
* 🚢 Ship
* 🚚 Truck

Each image is resized/prepared to the model's required input size of **32 × 32 pixels**.

---

## 🧠 CNN Architecture

The model is built using **PyTorch**.

### Convolutional Layers

```text
Input
  ↓
Conv2D (3 → 32)
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2D (32 → 64)
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2D (64 → 128)
  ↓
ReLU
  ↓
MaxPool
```

### Fully Connected Layers

```text
128 × 4 × 4
     ↓
Linear → 256
     ↓
ReLU
     ↓
Linear → 10
     ↓
Output
```

The final layer produces **10 output values**, one for each CIFAR-10 class.

---

## 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **Streamlit**
* **Pillow**
* **CIFAR-10 Dataset**

---

## ✨ Features

* 📤 Upload an image
* 🖼️ Display uploaded image
* 🔍 Predict image category
* 📊 Display prediction confidence
* 📋 Display supported categories
* 🌐 Streamlit web interface
* 🤖 PyTorch CNN model
* 💾 Loads trained model weights from `best_model.pth`

---

## 📁 Project Structure

```text
CIFAR10_CNN/
│
├── app.py
├── model.py
├── best_model.pth
├── requirements.txt
└── README.md
```

### File Description

| File               | Description                                    |
| ------------------ | ---------------------------------------------- |
| `app.py`           | Streamlit web application and prediction logic |
| `model.py`         | CNN architecture                               |
| `best_model.pth`   | Saved trained model weights                    |
| `requirements.txt` | Required Python dependencies                   |
| `README.md`        | Project documentation                          |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the project directory

```bash
cd CIFAR10_CNN
```

### 3. Create a virtual environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔮 How to Use

1. Open the Streamlit application.
2. Upload an image using the upload button.
3. The uploaded image will be displayed.
4. Click **Predict Category**.
5. The CNN will process the image.
6. The predicted category will be displayed.
7. The confidence score will also be shown.

Example:

```text
Predicted Category: Dog

Confidence: 98.21%
```

---

## 🔄 Image Preprocessing

Before sending an image to the CNN, the following preprocessing steps are applied:

```python
transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
```

### Steps

**Resize**

Converts the uploaded image to:

```text
32 × 32
```

**ToTensor**

Converts the image into a PyTorch tensor.

The image is represented in the format:

```text
[Channels, Height, Width]
```

For an RGB image:

```text
[3, 32, 32]
```

**Normalize**

Normalizes the image using the same normalization applied during model training.

---

## 🤖 Model Prediction

The model produces 10 output values called **logits**.

The predicted class is obtained using:

```python
predicted_class = torch.argmax(outputs, dim=1).item()
```

The logits are converted into probabilities using:

```python
probabilities = torch.softmax(outputs, dim=1)
```

The probability of the predicted class is then displayed as the model's confidence.

---

## 💾 Model Saving

During training, the model with the best validation loss is saved:

```python
torch.save(
    model.state_dict(),
    "best_model.pth"
)
```

The saved weights are loaded by the Streamlit application:

```python
model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location="cpu"
    )
)
```

This allows the deployed application to perform predictions without retraining the CNN.

---

## 🌐 Deployment

The application can be deployed using **Streamlit Community Cloud**.

Deployment workflow:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Deploy app.py
       ↓
Live Web Application
```

Make sure the repository contains:

```text
app.py
model.py
best_model.pth
requirements.txt
```

---

## ⚠️ Important Note

This model is trained specifically for the **10 CIFAR-10 categories**.

If an image outside these categories is uploaded, the model may still assign it to one of the 10 known classes because the classifier always chooses one of its available output classes.

Therefore, the displayed confidence should not be interpreted as a guarantee that the uploaded image actually belongs to the CIFAR-10 dataset.

---

## 📚 Learning Outcomes

Through this project, I learned and implemented:

* Convolutional Neural Networks
* `Conv2d`
* ReLU activation
* Max Pooling
* Flattening CNN feature maps
* Fully Connected Layers
* Cross Entropy Loss
* Adam Optimizer
* Training and Validation
* Model Evaluation
* Saving and Loading PyTorch Models
* Image Preprocessing
* Softmax Probabilities
* Streamlit Web Application Development
* Model Deployment

---

## 🔮 Future Improvements

Some possible improvements for this project:

* [ ] Add Top-3 predictions
* [ ] Improve UI/UX
* [ ] Add prediction history
* [ ] Add confidence visualization
* [ ] Add better handling for out-of-category images
* [ ] Improve model accuracy
* [ ] Add model performance metrics
* [ ] Deploy with a public live URL

---

## 👨‍💻 Author

**Shrajal Sahu**

Aspiring **ML / GenAI & Backend Developer**

### Skills & Interests

* Machine Learning
* Deep Learning
* PyTorch
* Generative AI
* NLP
* Backend Development
* FastAPI
* LangChain / LangGraph

---

## ⭐ If you found this project useful

Feel free to ⭐ the repository and explore the project!
