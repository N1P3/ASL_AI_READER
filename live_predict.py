import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

IMG_SIZE = 64
ROI_SIZE = 200

# === Załaduj model i klasy ===
model = load_model("model.h5")
classes = np.load("classes.npy")

# === MediaPipe Hands ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# === Kamera ===
cap = cv2.VideoCapture(2)
print("Naciśnij Q aby wyjść")

def extract_features(landmarks_xyz):
    landmarks = np.array(landmarks_xyz).reshape((21, 3))
    center = landmarks[0]
    norm_landmarks = landmarks - center

    fingertip_idxs = [4, 8, 12, 16, 20]
    dists = []
    for i in range(len(fingertip_idxs)):
        for j in range(i + 1, len(fingertip_idxs)):
            d = np.linalg.norm(landmarks[fingertip_idxs[i]] - landmarks[fingertip_idxs[j]])
            dists.append(d)

    norm_flat = norm_landmarks.flatten()
    features = np.concatenate([norm_flat, dists])
    return features

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nie udało się odczytać z kamery.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Rysuj landmarki
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            # Środek dłoni
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            cx = int(np.mean(x_coords) * w)
            cy = int(np.mean(y_coords) * h)

            half_size = ROI_SIZE // 2
            x_min = max(cx - half_size, 0)
            y_min = max(cy - half_size, 0)
            x_max = min(cx + half_size, w)
            y_max = min(cy + half_size, h)

            # Korekta krawędzi
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
            image_input = np.expand_dims(roi_norm, axis=0)

            # === Landmarky z bieżącej dłoni ===
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            if len(landmark_list) != 63:
                continue  # pomiń jeśli niekompletne

            features = extract_features(landmark_list)
            landmark_input = np.expand_dims(np.array(features, dtype="float32"), axis=0)

            # === Predykcja ===
            prediction = model.predict([image_input, landmark_input], verbose=0)[0]
            top_indices = prediction.argsort()[-3:][::-1]  # top 3 indeksy

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Wyświetl top 3 litery z prawdopodobieństwami
            for i, idx in enumerate(top_indices):
                letter = classes[idx]
                confidence = prediction[idx] * 100
                y_text = y_min - 10 - i * 30
                cv2.putText(frame, f'{letter} ({confidence:.1f}%)',
                            (x_min, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Tłumacz Migowy", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
