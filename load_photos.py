import os
import cv2
import numpy as np
import mediapipe as mp

# === Konfiguracja ===
IMG_SIZE = 64
DATASET_DIR = "asl_alphabet_train"
SAVE_IMAGES = False        # Czy zapisywać X.npy
SAVE_LANDMARKS = True      # Czy zapisywać X_landmarks.npy
SAVE_LABELS = True         # Czy zapisywać y.npy

X = []
y = []
X_landmarks = []

# === Inicjalizacja MediaPipe Hands (zwiększona dokładność) ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1  # lub 2 dla jeszcze dokładniejszego
)


def extract_features(landmarks_xyz):
    landmarks = np.array(landmarks_xyz).reshape((21, 3))

    # Środek dłoni – punkt 0
    center = landmarks[0]

    # Znormalizowane landmarki względem środka dłoni
    norm_landmarks = landmarks - center

    # Wybrane punkty palców (czubki): kciuk (4), wskazujący (8), środkowy (12), serdeczny (16), mały (20)
    fingertip_idxs = [4, 8, 12, 16, 20]
    dists = []
    for i in range(len(fingertip_idxs)):
        for j in range(i + 1, len(fingertip_idxs)):
            d = np.linalg.norm(landmarks[fingertip_idxs[i]] - landmarks[fingertip_idxs[j]])
            dists.append(d)

    # Spłaszczone cechy
    norm_flat = norm_landmarks.flatten()
    features = np.concatenate([norm_flat, dists])
    return features


for label in sorted(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(path):
        continue

    for file in os.listdir(path)[:2000]:
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # === Wykrywanie dłoni na oryginalnym obrazie ===
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
            continue  # brak dłoni = pomijamy

# === Konwersja do numpy ===
if SAVE_IMAGES:
    X = np.array(X, dtype="float32") / 255.0
if SAVE_LANDMARKS:
    X_landmarks = np.array(X_landmarks, dtype="float32")
if SAVE_LABELS:
    y = np.array(y)

# === Zapis wybranych danych ===
if SAVE_IMAGES:
    np.save("X.npy", X)
    print("Zapisano: X.npy")

if SAVE_LANDMARKS:
    np.save("X_landmarks.npy", X_landmarks)
    print("Zapisano: X_landmarks.npy")

if SAVE_LABELS:
    np.save("y.npy", y)
    print("Zapisano: y.npy")
