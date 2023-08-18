'''
Assignment 3: Detect faces in mask wearing images.
'''

# Import Libraries
import cv2

# Load Haar cascade classifiers for detecting frontal faces, eyes, mouths, and upper bodies
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
upper_body = cv2.CascadeClassifier('haarcascade_upperbody.xml')


# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 105

# Set up font and text properties for display
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (255, 255, 255)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "MASK"
not_weared_mask = "No MASK"

# Initialize the video capture
cap = cv2.VideoCapture(0)

while 1:
    # Capture each frame
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    # Convert Image into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the frame to grayscale
    (thresh, black_and_white) = cv2.threshold(
        gray, bw_threshold, 255, cv2.THRESH_BINARY)

    # Convert the grayscale frame to black and white using a threshold
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Detect faces in the black and white frame
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

    # Check if any faces are detected
    if (len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img, "No face found...", org, font, font_scale,
                    weared_mask_font_color, thickness, cv2.LINE_AA)
    elif (len(faces) == 0 and len(faces_bw) == 1):
        cv2.putText(img, weared_mask, org, font, font_scale,
                    weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        # Draw rectangle on face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect mouths in the grayscale frame
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)

        # If no mouth is detected, the person is wearing a mask
        if (len(mouth_rects) == 0):
            cv2.putText(img, weared_mask, org, font, font_scale,
                        weared_mask_font_color, thickness, cv2.LINE_AA)
        else:
            for (mx, my, mw, mh) in mouth_rects:

                if (y < my < y + h):
                    # If the mouth is detected within the face region, the person is not wearing a mask
                    cv2.putText(img, not_weared_mask, org, font, font_scale,
                                not_weared_mask_font_color, thickness, cv2.LINE_AA)

                    #cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                    break

    # Display the frame with the results
    cv2.imshow('Mask Detection', img)
    k = cv2.waitKey(30) & 0xff 
    if k == 27: # Press 'ESC' to exit the loop
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
