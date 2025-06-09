import time

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model

from Games.hangman import HangmanGame
from Games.snake import SnakeGame
from features import extract_features

mixed_precision.set_global_policy('mixed_float16')

IMG_SIZE = 64
ROI_SIZE = 200

model = load_model("model.h5")
classes = np.load("classes.npy")

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

output_path = "output.txt"
open(output_path, "w").close()

current_letter = ""
word_buffer = ""
space_start_time = None

snake_game_running = False
snake_game_window = None
snake_game_controller = None

hangman_game_running = False
hangman_game_window = None
hangman_game_controller = None

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Tłumacz Migowy")
app.geometry("900x700")

frame_label = ctk.CTkLabel(app, text="", width=640, height=480)
frame_label.pack(pady=10)

label_current = ctk.CTkLabel(app, text="Litera: ", font=("Arial", 20))
label_current.pack()

label_word = ctk.CTkLabel(app, text="", font=("Arial", 24, "bold"))
label_word.pack(pady=10)

cap = cv2.VideoCapture(0)


def launch_hangman_game():
    global hangman_game_running, hangman_game_window, hangman_game_controller
    if hangman_game_running:
        return

    hangman_game_window = ctk.CTkToplevel(app)
    hangman_game_window.title("Hangman")
    hangman_game_window.geometry("400x550")
    hangman_game_window.resizable(False, False)

    app_width = app.winfo_width()
    app_x = app.winfo_x()
    game_width = 400
    game_x = app_x + (app_width // 2) - (game_width // 2)
    game_y = 0
    hangman_game_window.geometry(f"+{game_x}+{game_y}")

    hangman_game_controller = HangmanGame(hangman_game_window, lambda: current_letter)
    hangman_game_running = True

def launch_snake_game():
    global snake_game_running, snake_game_window, snake_game_controller
    if snake_game_running:
        return

    snake_game_window = ctk.CTkToplevel(app)
    snake_game_window.title("Snake")
    snake_game_window.geometry("450x550")
    snake_game_window.resizable(False, False)

    app_width = app.winfo_width()
    app_x = app.winfo_x()
    game_width = 450
    game_x = app_x + (app_width // 2) - (game_width // 2)
    game_y = 50
    snake_game_window.geometry(f"+{game_x}+{game_y}")

    snake_game_controller = SnakeGame(snake_game_window, lambda: current_letter)
    snake_game_running = True


def update_frame():
    global current_letter, word_buffer, space_start_time, snake_game_running

    ret, frame = cap.read()
    if not ret:
        app.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    new_letter = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
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
            top_index = prediction.argmax()
            top_letter = classes[top_index]
            confidence = prediction[top_index] * 100

            if confidence > 80:
                new_letter = top_letter

    if snake_game_running and snake_game_controller:
        if new_letter in ["W", "A", "S", "D", "O"]:
            snake_game_controller.update_direction(new_letter)
            current_letter = ""

    if new_letter:
        if new_letter == "space":
            if space_start_time is None:
                space_start_time = time.time()
            elif time.time() - space_start_time >= 0.5 and word_buffer:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(word_buffer + " ")
                if word_buffer.lower() == "snake":
                    if not snake_game_running:
                        launch_snake_game()
                if word_buffer.lower() == "olek":
                    if not hangman_game_running:
                        launch_hangman_game()
                word_buffer = ""
                space_start_time = None
        else:
            current_letter = new_letter
            space_start_time = None
    else:
        space_start_time = None

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    ctk_img_for_label = ctk.CTkImage(light_image=img_pil, size=(frame.shape[1], frame.shape[0]))
    frame_label.configure(image=ctk_img_for_label)
    frame_label.image = ctk_img_for_label

    label_current.configure(text=f"Litera: {current_letter}")
    label_word.configure(text=word_buffer)

    app.after(10, update_frame)


def key_event(event):
    global word_buffer, snake_game_running
    if event.keysym == "Return" and current_letter and current_letter != "space":
        word_buffer += current_letter


app.bind("<Key>", key_event)
app.after(0, update_frame)
app.mainloop()

cap.release()
cv2.destroyAllWindows()
