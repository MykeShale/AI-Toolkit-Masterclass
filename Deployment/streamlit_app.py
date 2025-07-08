# 🖥️ Streamlit App: MNIST Digit Classifier

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

@st.cache_resource
def load_or_train_model():
    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Train the model (quick demo: use only a small subset)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:1000].reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train[:1000], 10)
    model.fit(x_train, y_train, epochs=2, batch_size=64)
    return model

model = load_or_train_model()

st.title("🧠 MNIST Handwritten Digit Classifier")
st.write("Upload a 28x28 grayscale digit image and get the predicted number!")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image.resize((28, 28)))
    image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

    st.image(image, caption="Uploaded Digit", width=150)
    prediction = model.predict(image_array)
    st.subheader(f"Prediction: {np.argmax(prediction)}")
    st.bar_chart(prediction[0])
