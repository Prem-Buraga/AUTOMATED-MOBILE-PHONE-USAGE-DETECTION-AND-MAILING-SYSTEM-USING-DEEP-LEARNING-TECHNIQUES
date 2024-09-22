# Automated Mobile Phone Usage Detection and Mailing System Using Deep Learning Techniques

## Overview
This project introduces a system utilizing YOLOv8 object detection for addressing concerns related to mobile phone incidents. Employing advanced computer vision cameras in targeted areas enables real-time identification of mobile phone usage. Upon detection, the system automates notifications, providing essential details - date, time, and captured images. Authorities have set fines for violations to discourage mobile phone usage in specified zones.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Flowchart](#flowchart)
- [Project Structure](#project-structure)
- [License](#license)

## Features
- **Real-time Detection**: Uses YOLOv8 for identifying mobile phone usage in restricted areas.
- **Email Notifications**: Sends automated emails with details of the violation to the concerned individuals.
- **Face Recognition**: Captures and recognizes faces of individuals using mobile phones.
- **Detailed Records**: Maintains logs of incidents with captured images and timestamps.

## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Automated-Mobile-Phone-Usage-Detection.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd Automated-Mobile-Phone-Usage-Detection
    ```
3. **Set up your Python environment**:
    - Ensure you have Python 3.x installed.
    - Install required libraries:
      ```bash
      pip install opencv-python numpy pandas face_recognition ultralytics openpyxl
      ```

## Usage
1. **Run the script**:
    ```bash
    python your_script.py
    ```
2. **Monitor the camera feed**: The system will continuously monitor for mobile phone usage.
3. **Receive notifications**: Upon detection, an email will be sent with the incident details.

## Dataset Preparation
To train the YOLO model effectively, you need a dataset consisting of images containing mobile phones in various scenarios. Follow these steps to prepare your dataset:

1. **Collect Images**: Gather images of mobile phones in different orientations, backgrounds, and lighting conditions. Aim for a diverse set of scenarios.

2. **Annotation**: Use a tool like [LabelImg](https://github.com/tzutalin/labelImg) to annotate images. Create bounding boxes around mobile phones and save the annotations in YOLO format.

3. **Organize the Dataset**: Structure your dataset into training and validation folders.

## Training
To train the YOLO model, follow these steps:

1. **Prepare the Dataset**: Ensure your images are annotated and organized correctly.

2. **Training the Model**: Use the YOLOv8 training script provided in the Ultralytics repository. Adjust the configuration file to point to your dataset and specify the number of classes.

3. **Save the Trained Model**: After training, save the model weights as `best.pt` for inference.

## Flowchart
Below is the flowchart illustrating the system's process for detecting mobile phone usage and sending notifications:

![image](https://github.com/user-attachments/assets/c83ac790-d2b0-4d8a-9762-1533e429a127) <!-- Replace with the actual path to your flowchart image -->

## Project Structure
- `your_script.py`: The main script handling detection and notifications.
- `best.pt`: Pre-trained YOLO model for mobile phone detection.
- `haarcascade_frontalface_default.xml`: Pre-trained Haar Cascade for face detection.
- `captured_faces/`: Folder for storing captured face images.
- `captured_frames/`: Folder for storing frames with detected mobile phones.
- `Details.xlsx`: Excel file containing student details for notification purposes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

