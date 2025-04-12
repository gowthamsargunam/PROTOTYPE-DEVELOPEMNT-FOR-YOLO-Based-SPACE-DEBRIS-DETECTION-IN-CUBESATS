import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import time
import base64

# Custom theme function
def set_bg_hack(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-attachment: fixed;
        }}
        /* Semi-transparent containers for better readability */
        .css-1aumxhk, .css-1v3fvcr, .css-1q8dd3e {{
            background-color: rgba(255, 255, 255, 0.8) !important;
            border-radius: 10px;
            padding: 20px;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Set our custom theme background
THEME_IMAGE_PATH = r"image.jpg" 
set_bg_hack(THEME_IMAGE_PATH)

# my yolo model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"yolov5\runs\train\exp\weights\best.pt")
    model.conf = 0.1  # Set default confidence threshold
    return model

model = load_model()

# Initialize detection logs and history
if 'logs' not in st.session_state:
    st.session_state.logs = []
    
if 'history' not in st.session_state:
    st.session_state.history = []
    
if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# Sidebar configuration
st.sidebar.title("Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold Level", 0.0, 1.0, 0.1)
input_type = st.sidebar.selectbox("Input Type", 
                                 ["Image", "Video", "Webcam"])

# History button in sidebar with anticlockwise symbol
if st.sidebar.button("⭯ HISTORY"):
    st.session_state.show_history = not st.session_state.show_history

# Clear history button in sidebar
if st.sidebar.button("Clear Detection History", help="Click to clear all detection logs"):
    if st.session_state.logs:  # Only save to history if there are logs to clear
        # Add deletion timestamp to each log entry
        deletion_time = time.strftime("%Y-%m-%d %H:%M:%S")
        deleted_logs = {
            "deletion_time": deletion_time,
            "logs": st.session_state.logs.copy()
        }
        st.session_state.history.append(deleted_logs)
    st.session_state.logs = []
    st.sidebar.success("Detection history cleared and saved to HISTORY!")

# Delete Permanently button in sidebar (new addition)
if st.sidebar.button("Delete Permanently", help="Permanently delete ALL logs and history (cannot be undone)"):
    st.session_state.logs = []
    st.session_state.history = []
    st.sidebar.error("All logs and history have been permanently deleted!")

# Main app layout
st.title("Space Debris Detection Dashboard")
st.caption("Real-time object detection and Alert System for Space Debris Monitoring")

# Alert section
alert_placeholder = st.empty()
log_placeholder = st.empty()

# Function to log detection events
def log_detection(label, confidence_score):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "Time": timestamp,
        "Object": label,
        "Confidence": f"{confidence_score:.2f}"
    }
    st.session_state.logs.append(log_entry)
    if len(st.session_state.logs) > 10:
        st.session_state.logs.pop(0)

# Image detection
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        results = model(image)
        
        # Process detections
        for detection in results.xyxy[0]:
            label = model.names[int(detection[5])]
            conf = float(detection[4])
            if conf >= confidence:
                log_detection(label, conf)
        
        # Display results
        st.image(np.squeeze(results.render()), 
                caption=f"Detected {len(results.xyxy[0])} objects")

# Video detection
elif input_type == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    
    if uploaded_file:
        # Save video to temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        frame_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            
            # Process detections
            for detection in results.xyxy[0]:
                label = model.names[int(detection[5])]
                conf = float(detection[4])
                if conf >= confidence:
                    log_detection(label, conf)
            
            # Display frame
            frame_placeholder.image(np.squeeze(results.render()), 
                                  channels="BGR")
        
        cap.release()

# Webcam detection
elif input_type == "Webcam":
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture webcam")
            break
        results = model(frame)
        
        # Process detections
        for detection in results.xyxy[0]:
            label = model.names[int(detection[5])]
            conf = float(detection[4])
            if conf >= confidence:
                log_detection(label, conf)
        
        # Display frame
        frame_placeholder.image(np.squeeze(results.render()), 
                              channels="BGR")
        
        # Exit on stop
        if st.button("Stop Webcam"):
            cap.release()
            break

# History panel
if st.session_state.show_history:
    with st.expander("HISTORY (Deleted Logs)", expanded=True):
        if st.session_state.history:
            for i, deleted_set in enumerate(reversed(st.session_state.history)):
                st.markdown(f"**Deleted on: {deleted_set['deletion_time']}**")
                st.table(deleted_set['logs'])
                
                # Add a divider between different deletion sets
                if i < len(st.session_state.history) - 1:
                    st.divider()
        else:
            st.info("No history available - no logs have been deleted yet")
            
        # Add a button to clear all history
        if st.button("Clear All History"):
            st.session_state.history = []
            st.rerun()

# Display alerts and logs
with alert_placeholder.container():
    st.subheader("Real-time Alerts")
    if st.session_state.logs:
        latest = st.session_state.logs[-1]
        st.warning(f"⚠️ Detection Alert! {latest['Object']} ({latest['Confidence']}) at {latest['Time']}")
    else:
        st.info("No active alerts")

with log_placeholder.container():
    st.subheader("Detection Log")
    if st.session_state.logs:
        # Add a clear button below the table
        col1, col2 = st.columns([3, 1])
        with col1:
            st.table(st.session_state.logs)
        with col2:
            if st.button("Clear Logs"):
                # Save to history before clearing
                deletion_time = time.strftime("%Y-%m-%d %H:%M:%S")
                deleted_logs = {
                    "deletion_time": deletion_time,
                    "logs": st.session_state.logs.copy()
                }
                st.session_state.history.append(deleted_logs)
                st.session_state.logs = []
                st.rerun()
    else:
        st.info("No detections yet")
