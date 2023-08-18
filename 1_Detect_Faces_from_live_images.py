'''
Assignment 1: Detect faces from live images taken from webcam.

This script will open a window that displays the live video feed from your webcam. 
Detected faces will be marked with a blue rectangle. 
To exit the loop and close the window, press the 'q' key.
Also, saves each detected face image as a separate file with a name like 'face_1.jpg', 'face_2.jpg', and so on
in an output directory called 'detected_faces' 
'''

import cv2
import os

# Load the Haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize and start realtime video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set video width
cap.set(4, 480)  # set video height

# Create a directory to save the detected face images
output_dir = 'detected_faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize face image count
face_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces and save the face images
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_count += 1
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_dir, f'face_{face_count}.jpg'), face_img)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

