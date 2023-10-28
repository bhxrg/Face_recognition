#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
'''if error appears as module not found then you shld first download any version of python and in command prompt: pip install opencv-python 
then itll download the dependencies then type python then import cv2 if no error its successfully installed to recheck type cv2.__version__
itll give the version name'''

# Load the pre-trained cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

# Check if the capture is already open; if so, close it
cap = cv2.VideoCapture(0)
if cap.isOpened():
    cap.release()

# Re-open the capture
cap = cv2.VideoCapture(0)

# Name the window and set a window property to bring it to the foreground and make it fullscreen
cv2.namedWindow('Face Detection', cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty('Face Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the frame with detected faces
    
    #count total number of faces
    cv2.putText(frame,'Number of Faces:' + str(len(faces)),(40,40),font,1,(255,0,0),2)
    cv2.imshow('Face detection',frame)

    # Break out of the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
else:
    # This will only execute if the loop ended without hitting a break statement.
    print("Live Face Capture Successful")

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




