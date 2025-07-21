import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# --- EfficientNet-B0 function ---
def create_efficientnet_model(num_classes=2):
    model = efficientnet_b0(weights=None) 
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2) 
)
    return model

# --- Load the trained model ---
@st.cache_resource
def load_model():
    model = create_efficientnet_model()
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- Preprocessing function ---
def preprocess_image(image: Image.Image):
    # Convert to RGB, resize, normalize, and convert to tensor
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = torch.tensor(img).unsqueeze(0)  # Add batch dimension
    return img

# --- Streamlit UI ---
st.title("MonReader: Page Flip Detection")

uploaded_file = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess and predict
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = "Page is being flipped" if predicted.item() == 1 else "Page is NOT being flipped"
    
    st.markdown(f"### Prediction: {label}")

    # show raw probabilities
    probs = torch.softmax(outputs, dim=1).numpy()[0]
    st.write(f"Confidence (Not Flip): {probs[0]:.2%}, (Flip): {probs[1]:.2%}")

st.markdown("---")
st.markdown("Developed using Streamlit and PyTorch.")