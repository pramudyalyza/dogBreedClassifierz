import streamlit as st

st.set_page_config(page_title="About - Dog Breeds Classifier", layout="wide")

st.title("📘 About the DogBreedClassifierz")
st.markdown("---")

st.markdown("""
## Model Description
This is a fine-tuned version of the **AlexNet** model, designed to classify images into one of **10 different dog breeds**.  
The original AlexNet architecture was pre-trained on the **ImageNet** dataset, and this version has been adapted using a custom dog breed dataset.

---

## Model Details

### Architecture
- **Base Model**: AlexNet  
- **Final Fully Connected Layer**: Modified to output 10 dog breed classes  
  - Input Features: 4096  
  - Output Classes: 10  

---

### Dataset
A custom dataset from [70 Dog Breeds - Kaggle](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set/data?select=dogs.csv) was used , containing images organized into **training**, **validation**, and **test** sets.
            
#### Dog Breeds Included:
1. Beagle  
2. Chihuahua  
3. Corgi  
4. Dalmation  
5. Doberman  
6. Golden Retriever  
7. Maltese  
8. Poodle  
9. Shiba Inu  
10. Siberian Husky  

#### Data Format
- **Image Format**: JPG  
- **Resolution**: 227x227 pixels (to match AlexNet’s input requirement)  
- **Structure**:  
  - `train/`: Training images  
  - `valid/`: Validation images  
  - `test/`: Test images  

---

## Training Setup
- **Optimizer**: SGD  
- **Learning Rate**: 0.001  
- **Momentum**: 0.9  
- **Loss Function**: CrossEntropyLoss  
- **Training Epochs**: 10  
- **Batch Size**: 32  

---
""")