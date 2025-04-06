# Brain Tumor Detection from MRI/CT Scans using YOLOv11 & Streamlit

## Description

This project provides an easy-to-use interface where users can upload an MRI or CT scan image (PNG, JPG, JPEG). A YOLOv11 model, trained specifically for brain tumor detection analyzes the image. If tumors are detected above a confidence threshold, they are highlighted with bounding boxes on the image. The application displays both the original and the processed image side-by-side and indicates if no tumors were detected. Users can also download the processed image.

---

## Features

* **Image Upload:** Supports PNG, JPG, and JPEG formats.
* **Tumor Detection:** Utilizes a YOLOv11 model to identify potential tumors.
* **Visual Feedback:** Displays the original image and the processed image with bounding boxes highlighting detected tumors.
* **Clear Results:** Explicitly states "No cancer detected" if the model finds no tumors above the confidence threshold.
* **Download Results:** Allows users to download the processed image with annotations.
* **User-Friendly Interface:** Built with Streamlit for simple interaction.
* **Image Resizing:** Automatically resizes large images to prevent memory issues while maintaining aspect ratio.
