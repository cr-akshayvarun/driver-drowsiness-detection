#!/usr/bin/env python
from imutils import face_utils
import imutils
import time
import dlib
import numpy as np
import cv2
import threading
from playsound import playsound

from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords

# -------------------- SETTINGS --------------------
EYE_AR_THRESH = 0.25



MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
ALARM_ON = False

# -------------------- ALARM FUNCTION --------------------
def play_alarm():
    try:
        playsound("alarm.mp3")
    except:
        print("[WARNING] Alarm sound not found!")

# -------------------- LOAD MODELS --------------------
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# -------------------- CAMERA (AUTO FIX) --------------------
print("[INFO] starting camera...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[INFO] Trying camera index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("[INFO] Trying camera index 2...")
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("[ERROR] No camera found!")
    exit()

time.sleep(2.0)

frame_height = 576

image_points = np.array([
    (359, 391), (399, 561), (337, 297),
    (513, 301), (345, 465), (453, 469)
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)

# -------------------- LOOP --------------------
while True:
    ret, frame = cap.read()

    # Safety check
    if not ret or frame is None:
        print("[ERROR] Camera frame not received")
        continue

    frame = imutils.resize(frame, width=1024, height=576)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    rects = detector(gray, 0)

    ear = 0
    mar = 0

    for rect in rects:
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # -------- EYE --------
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (350, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not ALARM_ON:
                    ALARM_ON = True
                    threading.Thread(target=play_alarm, daemon=True).start()
        else:
            COUNTER = 0
            ALARM_ON = False

        # -------- MOUTH --------
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning!", (800, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # -------- HEAD POSE --------
        for i, (x, y) in enumerate(shape):
            if i in [33, 8, 36, 45, 48, 54]:
                image_points[[33,8,36,45,48,54].index(i)] = (x, y)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        (head_tilt_degree, start_point, end_point, end_point_alt) = \
            getHeadTiltAndCoords(size, image_points, frame_height)

        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

        if head_tilt_degree:
            cv2.putText(frame, f'Head Tilt: {head_tilt_degree[0]}',
                        (170, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

    # -------- STATUS PANEL --------
    status = "NORMAL"
    color = (0, 255, 0)

    if COUNTER >= EYE_AR_CONSEC_FRAMES:
        status = "DROWSY"
        color = (0, 0, 255)
    elif mar > MOUTH_AR_THRESH:
        status = "YAWNING"
        color = (0, 165, 255)

    cv2.rectangle(frame, (0, 0), (320, 90), (50, 50, 50), -1)

    cv2.putText(frame, f"STATUS: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"MAR: {mar:.2f}", (160, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # -------- DISPLAY --------
    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------- CLEANUP --------------------
cap.release()
cv2.destroyAllWindows()