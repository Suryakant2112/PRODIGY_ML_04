import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# input gestures
GESTURES = {
    'f': 'fist',
    'p': 'palm',
    'o': 'ok'
}
DATA_PATH = "gestures.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# main part for collecting data;
# type the key and take the coordinates of the segment (x,y)
header = ['label'] + [f'p{i}_{axis}' for i in range(21) for axis in ['x', 'y']]
with open(DATA_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

# loop for taking ss of segment and storing it in csv
cap = cv2.VideoCapture(0)
print("Ready to collect data.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            key = cv2.waitKey(5) & 0xFF
            key_char = chr(key).lower()

            if key_char in GESTURES:
                gesture_name = GESTURES[key_char]
                print(f"Saving data for gesture: {gesture_name}")
                
                wrist_coords = hand_landmarks.landmark[0]
                landmarks_normalized = []
                for lm in hand_landmarks.landmark:
                    landmarks_normalized.extend([lm.x - wrist_coords.x, lm.y - wrist_coords.y])

                with open(DATA_PATH, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([gesture_name] + landmarks_normalized)

    cv2.imshow('Hand Gesture Data Collection', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
print(f"Data collection complete. Data saved to {DATA_PATH}")