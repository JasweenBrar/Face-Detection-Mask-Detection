'''
Assignment 2: Once you're done with images, then take short live video sequence and detect faces.

It is similar to the previous one, but it runs only for a specified duration (10 seconds in this case). 
The live video feed from your webcam will be displayed, and detected faces will be marked with a blue rectangle. 
To exit the loop and close the window early, press the 'q' key.
This script saves the output video with the detected faces as output.avi in the current directory.
'''

import cv2
import time

# Load the Haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize and start realtime video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set video width
cap.set(4, 480)  # set video height

# Set the duration (in seconds) for the video capture
duration = 10
end_time = time.time() + duration

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while time.time() < end_time:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture, the output video, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
