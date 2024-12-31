import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os

# Load TFLite model and allocate tensors
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# Preprocess frame for TFLite model
def preprocess_frame(frame, input_shape):
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    input_data = np.expand_dims(normalized_frame, axis=0).astype(np.float32)
    return input_data

# Perform prediction using TFLite model
def predict(frame, interpreter, input_details, output_details):
    input_shape = input_details[0]['shape']
    input_data = preprocess_frame(frame, input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    return prediction

# Annotate frame based on prediction
def annotate_frame(frame, prediction, threshold=0.5):
    label = "Violence" if prediction[0] > threshold else "Non-Violence"
    confidence = prediction[0] if label == "Violence" else 1 - prediction[0]
    color = (0, 255, 0) if label == "Violence" else (255, 0, 0)
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (10, 50), (300, 150), color, 3)
    return frame, label

# Streamlit UI
st.title("Violence Detection System")
st.write("Upload a video to detect violence or non-violence scenes.")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
model_path = st.text_input("Enter the path to your unquant.tflite model", "model/unquant.tflite")

if uploaded_video and model_path:
    # Load TFLite model
    interpreter, input_details, output_details = load_model(model_path)
    st.success("Model loaded successfully!")

    # Save uploaded video to temporary file
    temp_dir = tempfile.TemporaryDirectory()
    video_path = os.path.join(temp_dir.name, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open the video.")
    else:
        stframe = st.empty()
        st.write("Processing video... Press 'Stop' to end.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict and annotate frame
            prediction = predict(frame, interpreter, input_details, output_details)
            frame, label = annotate_frame(frame, prediction)

            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

        cap.release()
        temp_dir.cleanup()
