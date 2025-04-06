import streamlit as st
from PIL import Image
from io import BytesIO
import os
import traceback
import time
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(layout="wide", page_title="Brain Tumor Detection from MRI/CT Data")

st.write("## Brain Tumor Detection from MRI/CT Data")
st.write(
    ":brain: Upload your MRI or CT image to detect potential brain tumors. Our deep learning model analyzes the image and highlights abnormal regions. Full quality results and processed images can be downloaded from the sidebar. "
)
st.sidebar.write("## Upload and download :gear:")

# Increased file size limit
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Max dimensions for processing
MAX_IMAGE_SIZE = 640  # pixels

# --- IMPORTANT: Ensure this path is correct for your system ---
# Using a raw string literal (r'...') is good practice for Windows paths
try:
    Valid_model = YOLO(r'runs\detect\train9\weights\best.pt')
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.error("Please ensure the model path 'runs\\detect\\train9\\weights\\best.pt' is correct relative to where you run Streamlit.")
    st.stop() # Stop execution if model can't load
# --------------------------------------------------------------


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# Resize image while maintaining aspect ratio
def resize_image(image, max_size):
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return image.resize((new_width, new_height), Image.LANCZOS)

# Modified process_image function
@st.cache_data # Caching is useful for performance
def process_image(image_bytes):
    CONF_THRESHOLD = 0.25 # Minimum confidence score to draw a box
    BOX_COLOR = (0, 255, 0) # Green color for bounding boxes
    BOX_THICKNESS = 2
    has_detections = False # Initialize detection flag

    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB') # Ensure image is RGB
        resized = resize_image(image, MAX_IMAGE_SIZE)

        # Convert PIL image to NumPy array for YOLO/OpenCV
        # YOLO expects BGR by default if using OpenCV backend
        img_np = np.array(resized)
        # If your original PIL image wasn't RGB, conversion might be needed:
        # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Uncomment if needed

        # Process the image
        # Pass the NumPy array directly
        result_predict = Valid_model.predict(source=img_np, imgsz=(640), iou=0.4, conf=CONF_THRESHOLD, verbose=False)
        results = result_predict[0]

        # Start with a copy of the *input* numpy array for drawing
        # results.orig_img might already be BGR if YOLO used OpenCV
        img_to_draw_on = results.orig_img.copy()

        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy() # Get boxes as numpy array
            if len(boxes) > 0: # Check if any boxes were detected *above the confidence threshold*
                has_detections = True # Set flag to True
                for box_data in boxes:
                    coords = box_data.xyxy[0].astype(int) # [xmin, ymin, xmax, ymax]
                    xmin, ymin, xmax, ymax = coords
                    # Draw rectangle on the BGR image
                    cv2.rectangle(img_to_draw_on, (xmin, ymin), (xmax, ymax), BOX_COLOR, BOX_THICKNESS)
            # If len(boxes) == 0, has_detections remains False, and no boxes are drawn
        # else: results.boxes was None, so has_detections remains False

        # Convert the final image (with or without boxes) back to RGB for displaying in Streamlit
        plot_rgb = cv2.cvtColor(img_to_draw_on, cv2.COLOR_BGR2RGB)

        # Return the original resized PIL image, the processed NumPy array (now RGB), and the detection flag
        return resized, plot_rgb, has_detections

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        print(f"Error details in process_image: {traceback.format_exc()}") # Print full traceback to console
        return None, None, False # Return False for detections on error

# Modified fix_image function
def fix_image(upload):
    try:
        start_time = time.time()
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        status_text.text("Loading image...")
        progress_bar.progress(10)

        # Read image bytes
        if isinstance(upload, str):
            if not os.path.exists(upload):
                st.error(f"Default image not found at path: {upload}")
                return
            with open(upload, "rb") as f:
                image_bytes = f.read()
        else:
            image_bytes = upload.getvalue()

        status_text.text("Processing image...")
        progress_bar.progress(30)

        # Process image and get the detection status
        # Unpack the tuple: original resized PIL, processed numpy (RGB), boolean flag
        image, fixed_np, detections_found = process_image(image_bytes)

        # Check if processing failed (image or fixed_np will be None)
        if image is None or fixed_np is None:
            status_text.error("Image processing failed.")
            progress_bar.empty() # Clear progress bar on failure
            return # Exit the function

        progress_bar.progress(80)
        status_text.text("Displaying results...")

        # Display images
        col1.write("Original Image :camera:")
        col1.image(image) # Display original resized PIL image

        col2.write("Processed Image :wrench:")
        col2.image(fixed_np) # Display processed numpy array (RGB)

        # ---- Conditionally display "No cancer detected" ----
        if not detections_found:
            col2.info("No cancer detected")
        # ----------------------------------------------------

        # Convert processed numpy array (RGB) back to PIL for download function
        fixed_pil = Image.fromarray(fixed_np)

        # Prepare download button
        st.sidebar.markdown("\n")
        st.sidebar.download_button(
            "Download processed image",
            convert_image(fixed_pil), # Use the PIL version for saving
            "processed.png",
            "image/png"
        )

        progress_bar.progress(100)
        processing_time = time.time() - start_time
        status_text.text(f"Completed in {processing_time:.2f} seconds")

    except Exception as e:
        st.error(f"An error occurred in fix_image: {str(e)}")
        st.sidebar.error("Failed to process image")
        # Log the full error for debugging
        print(f"Error in fix_image: {traceback.format_exc()}")
        # Clean up UI elements on error
        if 'status_text' in locals() and status_text is not None: status_text.error("An error occurred.")
        if 'progress_bar' in locals() and progress_bar is not None: progress_bar.empty()


# --- UI Layout and Execution Logic ---
col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an MRI/CT image", type=["png", "jpg", "jpeg"])

# Process the image (either uploaded or default)
if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error(f"The uploaded file is too large. Please upload an image smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB.")
    else:
        with col1: # Clear previous images before processing new one
             st.empty()
        with col2:
             st.empty()
        fix_image(upload=my_upload)
else:
    # Display placeholder or instruction if no default image is processed yet
    col1.write("Original Image :camera:")
    col1.info("Upload an image using the sidebar to see the original.")
    col2.write("Processed Image :wrench:")
    col2.info("Upload an image using the sidebar to see the processed result.")

    # Optionally, process a default image if you uncomment this section
    # default_image_path = "./brainTumor.png" # Define your default image path
    # if 'default_processed' not in st.session_state: # Process default only once
    #     if os.path.exists(default_image_path):
    #          st.info("Processing default image...")
    #          fix_image(default_image_path)
    #          st.session_state.default_processed = True
    #     else:
    #          st.warning(f"Default image '{default_image_path}' not found.")