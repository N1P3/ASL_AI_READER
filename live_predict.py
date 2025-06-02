import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from features import extract_features
from tensorflow.keras import mixed_precision
import time

mixed_precision.set_global_policy('mixed_float16')

IMG_SIZE = 64
ROI_SIZE = 200

model = load_model("model.h5")
classes = np.load("classes.npy")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
print("Naciśnij Q aby wyjść")

word_buffer = ""
current_letter = ""
output_path = "output.txt"
open(output_path, "w").close()

space_timer_start = None
SPACE_HOLD_DURATION = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            cx = int(np.mean(x_coords) * w)
            cy = int(np.mean(y_coords) * h)

            half_size = ROI_SIZE // 2
            x_min = max(cx - half_size, 0)
            y_min = max(cy - half_size, 0)
            x_max = min(cx + half_size, w)
            y_max = min(cy + half_size, h)

            if x_max - x_min < ROI_SIZE:
                if x_min == 0:
                    x_max = ROI_SIZE
                elif x_max == w:
                    x_min = w - ROI_SIZE
            if y_max - y_min < ROI_SIZE:
                if y_min == 0:
                    y_max = ROI_SIZE
                elif y_max == h:
                    y_min = h - ROI_SIZE

            roi = frame[y_min:y_max, x_min:x_max]
            if roi.shape[0] != ROI_SIZE or roi.shape[1] != ROI_SIZE:
                continue

            roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi_norm = roi_resized / 255.0
            image_input = np.expand_dims(roi_norm, axis=0).astype('float16')

            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            if len(landmark_list) != 63:
                continue

            features = extract_features(landmark_list)
            landmark_input = np.expand_dims(np.array(features, dtype="float32"), axis=0)

            prediction = model.predict([image_input, landmark_input], verbose=0)[0]
            top_indices = prediction.argsort()[-3:][::-1]

            predicted_letter = classes[top_indices[0]]
            confidence = prediction[top_indices[0]] * 100

            if confidence > 80:
                current_letter = predicted_letter
            else:
                current_letter = ""

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            for i, idx in enumerate(top_indices):
                letter = classes[idx]
                conf = prediction[idx] * 100
                y_text = y_min - 10 - i * 30
                cv2.putText(frame, f'{letter} ({conf:.1f}%)',
                            (x_min, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if current_letter == "space":
        if space_timer_start is None:
            space_timer_start = time.monotonic()
        elif time.monotonic() - space_timer_start >= SPACE_HOLD_DURATION:
            if word_buffer:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(word_buffer + " ")
                word_buffer = ""
            space_timer_start = None  # reset timer after trigger
    else:
        space_timer_start = None

    cv2.putText(frame, word_buffer, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Litera: {current_letter}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Tłumacz Migowy", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:
        if current_letter and current_letter != "space":
            word_buffer += current_letter
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
