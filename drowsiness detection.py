import cv2  # type: ignore
import os
from keras.models import load_model  # type: ignore
import numpy as np
from pygame import mixer  # type: ignore
import time

# Initialize sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascade classifiers
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Load pre-trained model
model = load_model('models/cnncat2.h5')

# Labels for prediction
lbl = ['Closed', 'Open']

# Initialize video capture
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes
    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Draw black rectangle at the bottom
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Process right eye
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255.0
        r_eye = r_eye.reshape(1, 24, 24, -1)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        if rpred[0] == 1:
            lbl = 'Open'
        else:
            lbl = 'Closed'
        break

    # Process left eye
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255.0
        l_eye = l_eye.reshape(1, 24, 24, -1)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        if lpred[0] == 1:
            lbl = 'Open'
        else:
            lbl = 'Closed'
        break

    # Update score based on predictions
    if rpred[0] == 0 and lpred[0] == 0:  # Both eyes closed
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:  # At least one eye open
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    score = max(score, 0)  # Ensure score doesn't go below 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Trigger alarm if score exceeds threshold
    if score > 15:
        cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass

        thicc = thicc + 2 if thicc < 16 else thicc - 2
        thicc = max(thicc, 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
