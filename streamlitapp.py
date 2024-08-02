import streamlit as st
import os
from PIL import Image
import torch
import cv2
from pathlib import Path
from detect_dual import run as yolo_run_detection

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def add_logo(logo_path, size=(200, 150)):
    logo = Image.open(logo_path)
    logo = logo.resize(size)
    st.image(logo, use_column_width=False)

def run_detection(source_path):
    output_dir = Path("runs/detect/exp")
    yolo_run_detection(
        weights="models/detect/yolov9tr.pt",  # Adjust this path to your model weights
        source=source_path,
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=output_dir.parent,
        name=output_dir.name,
        exist_ok=True,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1,
    )
    output_path = output_dir / Path(source_path).name
    return str(output_path)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(0, frame_count, 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_path = f"temp_frame_{i}.jpg"
            cv2.imwrite(frame_path, frame)
            output_frame = run_detection(frame_path)
            yield output_frame
            os.remove(frame_path)  # Clean up temporary frame file
        else:
            break
    
    cap.release()

def main():
    st.title("YOLO Object Detection")
    
    add_logo("logo_ai.jpg")
    
    source_type = st.radio("Select source type:", ("Image", "Video"))
    
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
                try:
                    with st.spinner("Running detection..."):
                        output_frames = process_video(source_path)
                        result_placeholder = st.empty()
                        for frame in output_frames:
                            result_placeholder.image(frame, caption="Detection Result", use_column_width=True)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    if os.path.exists(source_path):
                        os.remove(source_path)  # Clean up temporary video file

if __name__ == "__main__":
    main()