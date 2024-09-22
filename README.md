# AUTOMATED MOBILE PHONE USAGE DETECTION AND MAILING SYSTEM USING DEEP LEARNING TECHNIQUES

## Overview
This project implements a system utilizing YOLOv8 object detection to address concerns related to mobile phone usage in restricted areas. Advanced computer vision cameras identify mobile phone usage in real-time. Upon detection, the system automates notifications, providing essential details such as date, time, and captured images. A fine system is enforced for violations to discourage mobile phone usage in designated zones.

## Table of Contents
- [Technologies](#technologies)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Flow Chart](#flow-chart)
- [Project Structure](#project-structure)
- [License](#license)

## Technologies
- Python
- YOLOv8
- OpenCV
- Face Recognition
- Pandas
- smtplib for email notifications
- openpyxl for Excel file handling

## Features
- **Real-time Mobile Detection**: Utilizes YOLOv8 to detect mobile phone usage.
- **Automated Notifications**: Sends email notifications with details of the violation.
- **Face Detection**: Captures faces associated with detected mobile phones for accountability.
- **Fine Management**: Updates fines in an Excel sheet for detected violations.

## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/AUTOMATED-MOBILE-PHONE-USAGE-DETECTION.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd AUTOMATED-MOBILE-PHONE-USAGE-DETECTION
    ```

3. **Set up your Python environment**:
    - Ensure you have Python 3.x installed.
    - Install required libraries:
      ```bash
      pip install opencv-python numpy face_recognition ultralytics pandas openpyxl
      ```

4. **Download Haar Cascade file**:
    - Place `haarcascade_frontalface_default.xml` in the project directory. You can download it from [here](https://github.com/opencv/opencv/tree/master/data/haarcascades).

5. **Prepare the YOLO model**:
    - Ensure you have the `best.pt` YOLO model file in the project directory.

## Usage
1. **Run the detection script**:
    ```bash
    python your_script_name.py
    ```
   Replace `your_script_name.py` with the actual filename of your script.

2. **Capture Mobile Usage**:
   - The system will use your webcam to monitor for mobile phone usage and capture faces.

3. **Email Notifications**:
   - Ensure you have configured your sender email and password in the script.

## Flow Chart
![image](https://github.com/user-attachments/assets/c83ac790-d2b0-4d8a-9762-1533e429a127)


## Project Structure
- `your_script_name.py`: The main script handling mobile phone detection and email notifications.
- `haarcascade_frontalface_default.xml`: Pre-trained Haar Cascade for face detection.
- `best.pt`: YOLOv8 model file for mobile phone detection.
- `captured_faces/`: Directory for storing captured face images.
- `captured_frames/`: Directory for storing frames with detected mobile phones.
- `Details.xlsx`: Excel file containing student information for notifications.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
