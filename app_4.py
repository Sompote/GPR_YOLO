import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
import torch
import sys
import cv2
from pytube import YouTube
import tempfile

# Assuming detect_dual.py has a run_detection function
from detect_dual import run as yolo_run_detection

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
new_env = os.environ.copy()

def add_logo(logo_path, size=(200, 150)):
    logo = Image.open(logo_path)
    logo = logo.resize(size)
    st.image(logo, use_column_width=False)

def run_detection(source_path):
    # Directly call the function from detect_dual.py
    yolo_run_detection(
        source=source_path,
        imgsz=(640, 640),
        device="cpu",
        weights="models/detect/yolov9tr.pt",
        name="yolov9_c_640_detect",
        exist_ok=True)
    
    # Find the output image or video
    output_dir = "runs/detect/yolov9_c_640_detect"
    output_path = os.path.join(output_dir, os.path.basename(source_path))
    return output_path

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Process every 10th frame
    for i in range(0, frame_count, 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_path = f"temp_frame_{i}.jpg"
            cv2.imwrite(frame_path, frame)
            output_frame = run_detection(frame_path)
            yield output_frame
        else:
            break
    
    cap.release()

def download_youtube_video(url):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        stream.download(output_path=os.path.dirname(tmp_file.name), filename=os.path.basename(tmp_file.name))
        return tmp_file.name

def main():
    st.title("YOLO9tr Object Detection")
    
    # Add the research center logo at the top of the app
    add_logo("logo_ai.jpg")
    
    source_type = st.radio("Select source type:", ("Image", "Video", "YouTube Link"))
    
    if source_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            source_path = "temp_image.jpg"
            with open(source_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            source_path = "United_States_000502.jpg"  # Default image
        
        st.image(source_path, caption="Image for Detection", use_column_width=True)
        
        if st.button("Run Detection"):
            with st.spinner("Running detection..."):
                output_path = run_detection(source_path)
                st.image(output_path, caption="Detection Result", use_column_width=True)
    
    elif source_type == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            source_path = "temp_video.mp4"
            with open(source_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Run Detection"):
                with st.spinner("Running detection..."):
                    output_frames = process_video(source_path)
                    for frame in output_frames:
                        st.image(frame, caption="Detection Result", use_column_width=True)
    
    elif source_type == "YouTube Link":
        youtube_url = st.text_input("Enter YouTube video URL:")
        if youtube_url:
            if st.button("Run Detection"):
                with st.spinner("Downloading video and running detection..."):
                    video_path = download_youtube_video(youtube_url)
                    output_frames = process_video(video_path)
                    for frame in output_frames:
                        st.image(frame, caption="Detection Result", use_column_width=True)
                    os.unlink(video_path)  # Clean up the temporary video file

if __name__ == "__main__":
    main()