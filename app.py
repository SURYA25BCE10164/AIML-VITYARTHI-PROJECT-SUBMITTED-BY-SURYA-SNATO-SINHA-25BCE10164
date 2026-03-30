import streamlit as st
from predict import predict_image
from PIL import Image
import tempfile

st.title("🍔 Food Image Classifier with Calories")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Prediction
    label, confidence, calorie = predict_image(file_path)

    st.success(f"Detected Food: {label}")
    st.info(f"Confidence: {confidence*100:.2f}%")
    st.warning(f"Estimated Calories: {calorie} kcal per serving")