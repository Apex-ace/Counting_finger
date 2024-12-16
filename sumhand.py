import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, x_max, y_min, y_max = 1000, 0, 1000, 0
            for landmark in hand_landmarks.landmark:
                x, y = landmark.x * image.shape[1], landmark.y * image.shape[0]
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            fingers = []
            tip_ids = [8, 12, 16, 20]
            for id in tip_ids:
                x, y = hand_landmarks.landmark[id].x * image.shape[1], hand_landmarks.landmark[id].y * image.shape[0]
                if y < hand_landmarks.landmark[id - 2].y * image.shape[0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            finger_count = sum(fingers)
            cv2.putText(image, f"Fingers: {finger_count}", (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Hand Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
