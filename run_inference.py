import cv2
import mediapipe as mp
import torch
import numpy as np

# reimport the model class to ensure its structure is known
from train_model import GestureClassifier 

# setup - model and cons load
MODEL_PATH = 'gesture_model.pth'
INPUT_SIZE = 42 # 21 landmarks * 2 coordinates

checkpoint = torch.load(MODEL_PATH, weights_only=False)
label_encoder_classes = checkpoint['label_encoder_classes']
NUM_CLASSES = len(label_encoder_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureClassifier(INPUT_SIZE, NUM_CLASSES).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# time for mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# loop for collecting segment and identifying its label
cap = cv2.VideoCapture(0)
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

            wrist_coords = hand_landmarks.landmark[0]
            landmarks_normalized = []
            for lm in hand_landmarks.landmark:
                landmarks_normalized.extend([lm.x - wrist_coords.x, lm.y - wrist_coords.y])
            
            input_tensor = torch.tensor([landmarks_normalized], dtype=torch.float32).to(device)

            # tensor to label
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_idx = torch.max(output, 1)
                predicted_gesture = label_encoder_classes[predicted_idx.item()]

            cv2.putText(image, predicted_gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()