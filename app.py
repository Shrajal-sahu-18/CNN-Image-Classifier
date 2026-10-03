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