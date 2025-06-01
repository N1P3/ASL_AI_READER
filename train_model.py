import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from tensorflow.keras import mixed_precision

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === Konfiguracja GPU ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Dynamiczne zarządzanie pamięcią GPU włączone")
    except RuntimeError as e:
        print(e)
else:
    print("Brak dostępnego GPU")

mixed_precision.set_global_policy('mixed_float16')

IMG_SIZE = 64

# === Wczytywanie danych ===
X = np.load("X.npy")
X_landmarks = np.load("X_landmarks.npy")
y = np.load("y.npy")

# === Kodyfikacja etykiet ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# === Mieszanie danych ===
X, X_landmarks, y_cat = shuffle(X, X_landmarks, y_cat, random_state=42)

# === Model dwuwejściowy ===

# Gałąź obrazu
image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
x = Conv2D(64, (3, 3), activation='relu', padding='same')(image_input)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)

# Gałąź landmarków
landmark_input = Input(shape=(X_landmarks.shape[1],), name="landmark_input")
y_land = Dense(128, activation='relu')(landmark_input)
y_land = Dropout(0.3)(y_land)

# Połączenie
combined = Concatenate()([x, y_land])
z = Dense(256, activation='relu')(combined)
z = Dropout(0.3)(z)

output = Dense(len(le.classes_), activation='softmax', dtype='float32')(z)

# === Kompilacja modelu ===
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
model.save("model.h5")  # zamiast save_format="keras" (niedostępne w TF 2.10)
np.save("classes.npy", le.classes_)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model zapisany jako model.h5, klasy jako classes.npy i label_encoder.pkl")
