# Car Detection using YOLOv2

This repository contains a Python implementation for detecting objects in images using a pre-trained YOLOv2 (You Only Look Once) model. The core logic is implemented in a Jupyter Notebook (`car_detection_using_yolo.ipynb`) and utilizes TensorFlow 2.x and Keras.

The model processes an input image and outputs the image with bounding boxes drawn around the detected objects, along with their class labels and confidence scores.

---

## 📂 Project Structure

<pre>
.
├── images/ # Input images for detection
├── model_data/ # Model files (weights, class names, anchors)
├── out/ # Output images with detected objects
├── yad2k/ # Helper scripts for YOLOv2 Keras conversion and utils
├── .gitignore # Git ignore file
├── Drive.ai Dataset Sample LICENSE # License for the sample dataset
├── LICENSE # Repository license
├── README.md # This README file
└── car_detection_using_yolo.ipynb # Main Jupyter Notebook with implementation
</pre>

---

## ⚙️ How It Works

The object detection pipeline is implemented in the `car_detection_using_yolo.ipynb` notebook and follows these key steps:

1.  **Load Model & Data**:
    *   A pre-trained YOLOv2 model is loaded from the `model_data/` directory.
    *   Class names are loaded from `model_data/coco_classes.txt`.
    *   Anchor box configurations are loaded from `model_data/yolo_anchors.txt`.

2.  **Image Preprocessing**:
    *   The input image is loaded and resized to the model's expected input size (608x608 pixels).
    *   Pixel values are normalized.

3.  **Inference**:
    *   The preprocessed image is passed through the YOLO model to get the raw feature map predictions.

4.  **Post-processing (`yolo_eval`)**:
    *   **Filter Boxes**: The raw output is decoded into bounding boxes. Boxes with a confidence score below a certain threshold (e.g., 0.6) are filtered out.
    *   **Non-Max Suppression (NMS)**: To avoid multiple detections for the same object, NMS is applied. It discards boxes that have a high Intersection over Union (IoU) with a higher-scoring box.
    *   **Scale Boxes**: The final bounding box coordinates are scaled back to the original image dimensions.

5.  **Visualization**:
    *   The final bounding boxes, along with class labels and scores, are drawn on the original image.
    *   The resulting image is saved to the `out/` directory.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.x installed. You will also need the following libraries:

*   TensorFlow
*   NumPy
*   Pillow (PIL)
*   Matplotlib
*   Pandas
*   SciPy

### Installation

1.  **Clone the repository:**
    ```sh
    git clone <url>
    ```

2.  **Install the required packages:**
    ```sh
    pip install tensorflow numpy pillow matplotlib pandas scipy
    ```

3.  **Download Pre-trained Model Weights:**
    This repository requires the pre-trained YOLOv2 model weights (`yolo.h5`). You need to download them and place the file inside the `model_data/` directory.

    You can find the original weights on the official YOLO website or convert them from the Darknet format using the scripts in the `yad2k/` directory.
    *   Official Site: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)

---

## USAGE

1.  **Add Input Images**: Place the images you want to process into the `images/` folder.

2.  **Run the Notebook**:
    *   Open and run the `car_detection_using_yolo.ipynb` Jupyter Notebook.
    *   The `predict()` function in the notebook will process the images, print the number of detected boxes, and display the output.

3.  **Check the Output**: The processed images with bounding boxes will be saved in the `out/` directory.

---

## 📄 License

The code in this repository is licensed under the terms of the [LICENSE](LICENSE) file.

The sample dataset images are provided under the `Drive.ai Dataset Sample LICENSE`.
