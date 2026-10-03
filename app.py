import torch
from torchvision import transforms
import streamlit as st 
from PIL import Image
from model import CNN

# Set Page Config
st.set_page_config(
    page_title="CIFAR-10 Image Clssifier",
    page_icon = "🖼️",
    layout = "centered"

)

st.title("🖼️ CIFAR-10 Image Classifier")
st.write("Upload an image and let the CNN predict its category.")

model = CNN()

## Load The Model
model.load_state_dict(torch.load("best_model.pth",map_location = "cpu"))

## Evaluation mode
model.eval()
