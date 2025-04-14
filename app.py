# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import streamlit as st

import torch
import requests
import torch.nn as nn
import streamlit as st

from PIL import Image
from io import BytesIO
from torchvision import models, transforms

# Load the model
model = models.alexnet(pretrained=False)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 10)
model.load_state_dict(torch.load('dogBreedsClassifier.pth', map_location=torch.device('cpu')))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Transformation
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Label mapping
selected_breeds = [
    "Beagle", "Chihuahua", "Corgi", "Dalmation", "Doberman",
    "Golden Retriever", "Maltese", "Poodle", "Shiba Inu", "Siberian Husky"
]
label_map = {label: idx for idx, label in enumerate(selected_breeds)}

# Prediction function
def predict_image(image):
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)
    predicted_label = [label for label, idx in label_map.items() if idx == predicted_class.item()][0]
    return predicted_label, confidence.item() * 100

# Streamlit UI
st.title("🐶 DogBreedClassifierz")
st.markdown("Upload an image, take a photo, or paste an image URL to predict the dog breed!")

option = st.radio("Choose image input method:", ["Upload", "Camera", "Image URL"])

image = None

if option == "Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif option == "Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image)

elif option == "Image URL":
    image_url = st.text_input("Paste the image URL here")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        except:
            st.error("⚠️ Failed to load image from URL.")

# Show and predict
if image:
    st.markdown("### Prediction")
    col1, col2 = st.columns([2, 1])  # wider image, narrower result

    with col1:
        st.image(image, caption="Your Image", use_column_width=True)

    predicted_label, confidence = predict_image(image)

    with col2:
        st.success(f"**Breed:** {predicted_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")