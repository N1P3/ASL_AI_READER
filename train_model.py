import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# === Parametry ===
IMG_SIZE = 64

# === Wczytaj dane ===
X = np.load("X.npy")                    # obrazy 64x64x3
X_landmarks = np.load("X_landmarks.npy")  # cechy (63 + 10 = 73)
y = np.load("y.npy")                   # etykiety tekstowe

# === Kodyfikacja etykiet ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Mieszamy dane
X, X_landmarks, y_cat = shuffle(X, X_landmarks, y_cat, random_state=42)

# === Model dwuwejściowy ===
# Wejście obrazu
image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Flatten()(x)

# Wejście cech landmarków (73 cechy)
landmark_input = Input(shape=(73,), name="landmark_input")
y_land = Dense(64, activation='relu')(landmark_input)

# Połączenie obu gałęzi
combined = Concatenate()([x, y_land])
z = Dense(128, activation='relu')(combined)
output = Dense(len(le.classes_), activation='softmax')(z)

# Kompilacja modelu
model = Model(inputs=[image_input, landmark_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Trenowanie ===
es = EarlyStopping(patience=3, restore_best_weights=True)
model.fit(
    [X, X_landmarks], y_cat,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[es]
)

# === Zapis modelu i klas ===
model.save("model.h5")
np.save("classes.npy", le.classes_)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model zapisany jako model.h5, klasy jako classes.npy i label_encoder.pkl")
