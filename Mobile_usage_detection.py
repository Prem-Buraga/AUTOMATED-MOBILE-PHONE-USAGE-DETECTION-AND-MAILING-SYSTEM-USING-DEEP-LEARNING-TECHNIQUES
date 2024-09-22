import cv2
import os
import numpy as np
import face_recognition
from ultralytics import YOLO
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import openpyxl

# Load the YOLO model for mobile phone detection
mobile_net = YOLO(r"best.pt")

# Load face recognition model (you need to have a pre-trained model, e.g., from OpenCV or dlib)
face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

# Create a folder to store captured faces
output_folder_faces = 'captured_faces'
output_folder_frames = 'captured_frames'
os.makedirs(output_folder_faces, exist_ok=True)
os.makedirs(output_folder_frames, exist_ok=True)

# Open a video capture object (you can replace 0 with the video file path)
cap = cv2.VideoCapture(0)

# Mobile detection variables
mobile_detected = False
mobile_coordinates = None


# Counter for captured faces
captured_faces = 0
captured_phones = 0
capture_limit = 2

# Declare the variable outside the loop
captured_face_encoding = None

# Load the Excel sheet into a DataFrame
excel_path = r'Details.xlsx'
df = pd.read_excel(excel_path, engine='openpyxl')

# Function to send an email with an attachment
def send_email_with_attachment(receiver_email, subject, message, attachment_path):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Attach the frame image to the email
    with open(attachment_path, 'rb') as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(attachment_path)}")
        msg.attach(part)

    # Connect to the email server and send the email
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

    print(f"Email sent to {email}")


# Flag to indicate if a match has been found
match_found = False

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not mobile_detected:
        # Perform mobile phone detection using YOLO
        mobile_results = mobile_net(frame)

        for results in mobile_results:
            # Access the attributes of the Results object
            boxes = results.boxes  # Access the Boxes object

            # Check if there are any detections
            if boxes is not None and len(boxes.data) > 0:
                # Access the coordinates of the first detection
                coordinates = boxes.xyxy[0].tolist()  # Assuming there is at least one detection
                # print("Coordinates:", coordinates)

                # Extract individual coordinates
                x_min, y_min, x_max, y_max = coordinates[:4]

                confidence = boxes.data[0, 4]
                class_name = mobile_net.names[int(boxes.data[0, 5])]

                if class_name == 'phone_call' and confidence > 0.5:
                    # Set the flag to capture the face nearest to the mobile phone
                    mobile_detected = True
                    mobile_coordinates = (x_min, y_min, x_max - x_min, y_max - y_min)
                    # print("Mobile Coordinates: ", mobile_coordinates)

                    # Draw bounding box for the detected mobile phone (in red)
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

    if mobile_detected:
        # Extend the search area around the mobile phone coordinates
        x, y, w, h = map(int, mobile_coordinates)
        search_margin = 250  # You can adjust this value based on your requirements
        x_search = max(0, x - search_margin)
        y_search = max(0, y - search_margin)
        w_search = min(frame.shape[1], w + 2 * search_margin)
        h_search = min(frame.shape[0], h + 2 * search_margin)
        roi_search = frame[y_search:y_search + h_search, x_search:x_search + w_search]

        # Check if the search area is not empty
        if roi_search.shape[0] > 0 and roi_search.shape[1] > 0:
            if captured_phones < capture_limit:
                # Save the frame with bounding boxes
                current_date_time = datetime.now().strftime("%d-%m-%Y_%Hhr_%Mmin")
                frame_filename_search = os.path.join(output_folder_frames,
                                                    f"{current_date_time}_{len(os.listdir(output_folder_frames)) + 1}.png")
                frame_filename = int(f"{len(os.listdir(output_folder_frames)) + 1}")
                # print("Frame file name : ", frame_filename)
                frame_filename_prev = f"{current_date_time}_{frame_filename - 1}"
                # print("frame_filename_prev : ",frame_filename_prev)
                frame_filename_attach = os.path.join(output_folder_frames,
                                                    f"{frame_filename_prev}.png")
                # print("frame_filename_attach : ",frame_filename_attach)
                # frame_filename_name, frame_filename_no = frame_filename.split('_')
                # print("frame_filename_name", frame_filename_name)
                # print("frame_filename_no", frame_filename_no)
                cv2.imwrite(frame_filename_search, frame)
                print(f"Frame saved: {frame_filename_search}")
                captured_phones += 1

                # Perform face detection in the extended search area
                gray_roi_search = cv2.cvtColor(roi_search, cv2.COLOR_BGR2GRAY)
                faces_search = face_cascade.detectMultiScale(gray_roi_search, scaleFactor=1.3, minNeighbors=5)

                # Find the face closest to the mobile phone
                if len(faces_search) > 0:
                    fx_search, fy_search, fw_search, fh_search = faces_search[0]

                    # Draw bounding box for the mobile phone (in red)
                    cv2.rectangle(frame, (x_search + fx_search, y_search + fy_search),
                                  (x_search + fx_search + fw_search, y_search + fy_search + fh_search), (255, 0, 0),
                                  2)

                    # Save the detected face to the output folder
                    face_image_search = frame[y_search + fy_search:y_search + fy_search + fh_search,
                                       x_search + fx_search:x_search + fx_search + fw_search]
                    face_filename_search = os.path.join(output_folder_faces,
                                                        f"face_{len(os.listdir(output_folder_faces)) + 1}.png")
                    # print("face_filename_search : ", face_filename_search)
                    cv2.imwrite(face_filename_search, face_image_search)
                    print(f"Face saved: {face_filename_search}")
                    captured_faces += 1

                    # Perform face matching with the captured face and images in the folder
                    if captured_faces >= capture_limit and not match_found:
                        
                        # Release the video capture object and close all windows
                        cap.release()
                        cv2.destroyAllWindows()

                        print("Performing Matching...")

                        # Load the captured face image
                        captured_face_path = face_filename_search
                        # print(captured_face_path)
                        captured_face_image = face_recognition.load_image_file(captured_face_path)
                        captured_face_encodings = face_recognition.face_encodings(captured_face_image)

                        # Check if at least one face is found
                        if captured_face_encodings:
                            captured_face_encoding = captured_face_encodings[0]
                            # Directory containing images with specific numbers
                            image_folder = r'Img_No'

                            # Iterate through images in the folder and perform face matching
                            for image_filename in os.listdir(image_folder):
                                image_path = os.path.join(image_folder, image_filename)

                                # Load the image for comparison
                                comparison_image = face_recognition.load_image_file(image_path)
                                comparison_encoding = face_recognition.face_encodings(comparison_image)

                                if len(comparison_encoding) > 0:
                                    
                                    comparison_encoding = comparison_encoding[0]

                                    # Compare the face encodings
                                    matches = face_recognition.compare_faces([captured_face_encoding], comparison_encoding, tolerance=0.5)

                                    # If a match is found, retrieve the specific number
                                    if any(matches):
                                        print(f"Match found for : {image_filename}")

                                        roll_number = int(image_filename.split('.')[0])  # Assuming filenames are numbers
                                        print(f"Match found for specific number: {roll_number}")
                                        # Perform further actions with the roll_number, e.g., storing in a variable

                                        # Retrieve information from the Excel sheet based on the roll number
                                        student_info = df[df['Roll Number'] == roll_number]

                                        if not student_info.empty:
                                            # Extract information from the DataFrame
                                            name = student_info.iloc[0]['Name']
                                            email = student_info.iloc[0]['Email']
                                            phone_number = student_info.iloc[0]['Phone Number']  # Assuming the column name in Excel is 'Phone'

                                            # Update the 'Fine' column in the Excel sheet
                                            df.loc[df['Roll Number'] == roll_number, 'Fine'] += 500
                                            df.to_excel(excel_path, index=False, engine='openpyxl')  # Save the updated DataFrame to the Excel file

                                            updated_fine = df.loc[df['Roll Number'] == roll_number, 'Fine' ].values[0]
                                            print("Updated fine : ", updated_fine)
                                            
                                            # Send email
                                            sender_email = '' #Enter your mail
                                            sender_password = '' # enter you app password
                                            receiver_email = email
                                            subject = 'Notice: Mobile Phone Usage Detected in Restricted Area'
                                            message = f"Dear {name},\n\nWe hope this message finds you well. We regret to inform you that our surveillance system has detected the usage of a mobile phone in a restricted area. As per our policy, a fine of INR 500 has been imposed.\n\nTo resolve this matter promptly, please submit the fine to the designated collection point within the next 48 Hours. Failure to comply may result in further actions.\n\nFor any clarification or dispute regarding this fine, please contact Us.\n\nTotal Fine Imposed : {updated_fine}\n\nThank you for your understanding and cooperation.\n"
                                            

                                        else:
                                            print(f"No information found for roll number: {roll_number}")

                                        # Set the flag to True
                                        match_found = True
                                        # Perform further actions with the roll_number, e.g., storing in a variable
                                        break
                                    else:
                                        print(f"No match found for {image_filename}")
                                else:
                                    print("Error: No face found in the captured image.")
                                    # Add appropriate handling for the case when no face is found, such as skipping to the next iteration
                                    continue
                        else :
                            print("Error: No faces detected in the captured face image.")                    
                        # Reset variables for the next capture
                        mobile_detected = False
                        mobile_coordinates = None
                        captured_faces = 0
                        captured_phones = 0

                        # if email == None:
                        # Send email with the captured frame as an attachment
                        send_email_with_attachment(email, subject, message, frame_filename_attach)

                    else :
                        # Handle the case when no faces are detected
                        print("Error: No faces detected in the captured image.")
                        continue
    # Display the frame
    cv2.imshow('Mobile and Face Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Break the loop if a match is found
    if match_found:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

print("Completed")