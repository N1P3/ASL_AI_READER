import os
import cv2
import numpy as np
import mediapipe as mp
from features import extract_features

# === Konfiguracja ===
IMG_SIZE = 64
DATASET_DIR = "ASL_Alphabet_Dataset/asl_alphabet_train"
SAVE_IMAGES = True
SAVE_LANDMARKS = True
SAVE_LABELS = True

X = []
y = []
X_landmarks = []

# === Inicjalizacja MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1  # Można ustawić na 2 dla jeszcze większej dokładności
)

for label in sorted(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(path):
        continue

    for file in os.listdir(path)[:8000]:  # można zwiększyć limit jeśli potrzeba
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Wykrywanie dłoni
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

            if len(landmarks) == 63:
                features = extract_features(landmarks)

                if SAVE_IMAGES:
                    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img_resized)

                if SAVE_LANDMARKS:
                    X_landmarks.append(features)

                if SAVE_LABELS:
                    y.append(label)

                print(f"Wykryto dłoń: {label}/{file}")
            else:
                print(f"Niekompletne landmarky: {label}/{file} – pomijam")
        else:
            print(f"Brak dłoni – pomijam: {label}/{file}")
            continue

# === Konwersja do NumPy ===
if SAVE_IMAGES:
    X = np.array(X, dtype="float32") / 255.0  # normalizacja obrazów
if SAVE_LANDMARKS:
    X_landmarks = np.array(X_landmarks, dtype="float32")
if SAVE_LABELS:
    y = np.array(y)

# === Zapis danych ===
if SAVE_IMAGES:
    np.save("X.npy", X)
    print("Zapisano: X.npy")

if SAVE_LANDMARKS:
    np.save("X_landmarks.npy", X_landmarks)
    print("Zapisano: X_landmarks.npy")

if SAVE_LABELS:
    np.save("y.npy", y)
    print("Zapisano: y.npy")
