import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the pre-trained model
model = tf.keras.models.load_model("./mnist_cnn_model.h5")

st.title("MNIST handwritten digit classification")
st.write("Upload an image of a handwritten digit (0-9) to get its prediction.")
# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors for better prediction
    image = image.resize((28, 28))  # Resize to 28x28   
    st.image(image, caption="Uploaded Image", width=150)

    img_array = np.array(image).reshape(1, 28, 28, 1)  # Reshape for model input
    img_array = img_array / 255.0  # Normalize the image
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction, axis=1)[0]  # Get the predicted digit
    st.write(f"Predicted Digit: {predicted_digit}")
else:
    st.write("Please upload an image to get a prediction.")