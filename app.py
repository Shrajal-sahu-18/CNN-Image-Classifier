import torch
from torchvision import transforms
import streamlit as st 
from PIL import Image
from model import CNN

st.set_page_config(
    page_title="CIFAR-10 Image Clssifier",
    page_icon = "🖼️",
    layout = "centered"

)

st.title("🖼️ CIFAR-10 Image Classifier")
st.write("Upload an image and let the CNN predict its category.")
## Model
model = CNN()

## Load The Model
model.load_state_dict(torch.load("best_model.pth",map_location = "cpu"))

## Evaluation mode
model.eval()

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
st.info( "Supported Categories: " + ", ".join(classes) )
# st.info("""
# This CNN model is trained on CIFAR-10 and can classify:
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
# """)

# st.write("Supported Category")
# st.write(",".join(classes))

## Transform and Normalize
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

uploaded_image = st.file_uploader(
    "📤 Upload an image",
    type = ["jpg","jpeg","png","webp"]
)

if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    st.image(
        image,
        caption = "Uploaded Image",
        width = 300
    )

    # Convert Image To Tensor
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    if st.button("🔍 Predict Category"):

        with torch.no_grad():
            outputs = model(image_tensor)

            predicted_class =torch.argmax(outputs,dim=1).item()

            predicted_label = classes[predicted_class]

            probabilities = torch.softmax(outputs,dim = 1)

            confidence = probabilities[0][predicted_class].item()

            st.success(f"Prediction :{predicted_label}")
            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )
