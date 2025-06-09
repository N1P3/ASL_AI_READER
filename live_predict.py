import time

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model

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

cap = cv2.VideoCapture(2)


class SnakeGame:
    def __init__(self, master, control_letter_callback):
        self.master = master
        self.control_letter_callback = control_letter_callback
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.score = 0
        self.game_over = False
        self.game_started = False
        self.board_size = 10
        self.tile_size = 40

        self.snake = [(5, 5)]
        self.snake_direction = (0, 1)
        self.food = self._generate_food()

        self._load_textures()

        self.canvas = ctk.CTkCanvas(self.master, width=self.board_size * self.tile_size,
                                    height=self.board_size * self.tile_size,
                                    bg="black", highlightthickness=0)
        self.canvas.pack(pady=10)

        self.score_label = ctk.CTkLabel(self.master, text=f"Punkty: {self.score}", font=("Arial", 24, "bold"))
        self.score_label.pack(pady=5)

        self._draw_board()
        self.game_loop_id = None
        self.countdown_val = 3
        self.start_countdown()

    def _load_textures(self):
        texture_paths = {
            "head_up": "textures/head_up.png",
            "head_down": "textures/head_down.png",
            "head_left": "textures/head_left.png",
            "head_right": "textures/head_right.png",
            "body_base": "textures/body_vertical.png",
            "tail_base": "textures/tail_up.png",
            "food": "textures/food.png",
        }

        self.head_ctk_images = {}
        self.body_ctk_images_oriented = {}
        self.tail_ctk_images_oriented = {}
        self.food_ctk_image = None

        self.head_photo_images = {}
        self.body_photo_images_oriented = {}
        self.tail_photo_images_oriented = {}
        self.food_photo_image = None

        def load_image_pil(path, size):
            try:
                return Image.open(path).resize((size, size), Image.Resampling.LANCZOS)
            except FileNotFoundError:
                print(f"Error: Texture file not found at {path}. Using fallback color.")
                return Image.new('RGBA', (size, size), (100, 100, 100, 255))
            except Exception as e:
                print(f"Error loading image {path}: {e}. Using fallback color.")
                return Image.new('RGBA', (size, size), (100, 100, 100, 255))


        pil_head_up = load_image_pil(texture_paths["head_up"], self.tile_size)
        pil_head_down = load_image_pil(texture_paths["head_down"], self.tile_size)
        pil_head_left = load_image_pil(texture_paths["head_left"], self.tile_size)
        pil_head_right = load_image_pil(texture_paths["head_right"], self.tile_size)

        self.head_ctk_images[(0, -1)] = ctk.CTkImage(light_image=pil_head_up, size=(self.tile_size, self.tile_size))
        self.head_photo_images[(0, -1)] = ImageTk.PhotoImage(pil_head_up)

        self.head_ctk_images[(0, 1)] = ctk.CTkImage(light_image=pil_head_down, size=(self.tile_size, self.tile_size))
        self.head_photo_images[(0, 1)] = ImageTk.PhotoImage(pil_head_down)

        self.head_ctk_images[(-1, 0)] = ctk.CTkImage(light_image=pil_head_left, size=(self.tile_size, self.tile_size))
        self.head_photo_images[(-1, 0)] = ImageTk.PhotoImage(pil_head_left)

        self.head_ctk_images[(1, 0)] = ctk.CTkImage(light_image=pil_head_right, size=(self.tile_size, self.tile_size))
        self.head_photo_images[(1, 0)] = ImageTk.PhotoImage(pil_head_right)


        base_body_pil = load_image_pil(texture_paths["body_base"], self.tile_size)
        base_tail_pil = load_image_pil(texture_paths["tail_base"], self.tile_size)


        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        angles = [0, 180, -90, 90]

        for direction, angle in zip(directions, angles):
            rotated_body_pil = base_body_pil.rotate(angle)
            self.body_ctk_images_oriented[direction] = ctk.CTkImage(light_image=rotated_body_pil,
                                                                    size=(self.tile_size, self.tile_size))
            self.body_photo_images_oriented[direction] = ImageTk.PhotoImage(rotated_body_pil)

            rotated_tail_pil = base_tail_pil.rotate(angle)
            self.tail_ctk_images_oriented[direction] = ctk.CTkImage(light_image=rotated_tail_pil,
                                                                    size=(self.tile_size, self.tile_size))
            self.tail_photo_images_oriented[direction] = ImageTk.PhotoImage(rotated_tail_pil)

        # Load food texture
        pil_food = load_image_pil(texture_paths["food"], self.tile_size)
        self.food_ctk_image = ctk.CTkImage(light_image=pil_food, size=(self.tile_size, self.tile_size))
        self.food_photo_image = ImageTk.PhotoImage(pil_food)

    def _generate_food(self):
        while True:
            food_x = np.random.randint(0, self.board_size)
            food_y = np.random.randint(0, self.board_size)
            if (food_x, food_y) not in self.snake:
                return (food_x, food_y)

    def _get_segment_direction(self, current_pos, prev_pos):
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]

        if abs(dx) > self.board_size / 2:
            dx = (dx + self.board_size) % self.board_size if dx < 0 else (dx - self.board_size) % self.board_size
        if abs(dy) > self.board_size / 2:
            dy = (dy + self.board_size) % self.board_size if dy < 0 else (dy - self.board_size) % self.board_size

        dx = int(np.sign(dx)) if dx != 0 else 0
        dy = int(np.sign(dy)) if dy != 0 else 0

        return (dx, dy)

    def _draw_board(self):
        self.canvas.delete("all")

        for x in range(self.board_size):
            for y in range(self.board_size):
                color = "gray10" if (x + y) % 2 == 0 else "gray15"
                self.canvas.create_rectangle(
                    x * self.tile_size, y * self.tile_size,
                    (x + 1) * self.tile_size, (y + 1) * self.tile_size,
                    fill=color, outline=""
                )

        if self.food_photo_image:
            self.canvas.create_image(
                self.food[0] * self.tile_size + self.tile_size // 2,
                self.food[1] * self.tile_size + self.tile_size // 2,
                image=self.food_photo_image, tags="food"
            )
        else:
            self.canvas.create_oval(
                self.food[0] * self.tile_size, self.food[1] * self.tile_size,
                (self.food[0] + 1) * self.tile_size, (self.food[1] + 1) * self.tile_size,
                fill="red", tags="food"
            )

        for i, segment in enumerate(self.snake):
            x_pos = segment[0] * self.tile_size
            y_pos = segment[1] * self.tile_size

            if i == 0:
                head_photo_image = self.head_photo_images.get(self.snake_direction)
                if head_photo_image:
                    self.canvas.create_image(
                        x_pos + self.tile_size // 2,
                        y_pos + self.tile_size // 2,
                        image=head_photo_image, tags="snake_head"
                    )
                else:
                    self.canvas.create_rectangle(
                        x_pos, y_pos,
                        x_pos + self.tile_size, y_pos + self.tile_size,
                        fill="green", tags="snake_head"
                    )
            elif i == len(self.snake) - 1:
                if len(self.snake) > 1:
                    prev_segment = self.snake[i - 1]
                    tail_direction = self._get_segment_direction(segment, prev_segment)
                    tail_photo_image = self.tail_photo_images_oriented.get(tail_direction)
                    if tail_photo_image:
                        self.canvas.create_image(
                            x_pos + self.tile_size // 2,
                            y_pos + self.tile_size // 2,
                            image=tail_photo_image, tags="snake_tail"
                        )
                    else:
                        self.canvas.create_rectangle(
                            x_pos, y_pos,
                            x_pos + self.tile_size, y_pos + self.tile_size,
                            fill="dark green", tags="snake_tail"
                        )
                else:
                    self.canvas.create_rectangle(
                        x_pos, y_pos,
                        x_pos + self.tile_size, y_pos + self.tile_size,
                        fill="dark green", tags="snake_tail"
                    )
            else:
                prev_segment = self.snake[i - 1]
                body_direction = self._get_segment_direction(segment, prev_segment)
                body_photo_image = self.body_photo_images_oriented.get(body_direction)
                if body_photo_image:
                    self.canvas.create_image(
                        x_pos + self.tile_size // 2,
                        y_pos + self.tile_size // 2,
                        image=body_photo_image, tags="snake_body"
                    )
                else:
                    self.canvas.create_rectangle(
                        x_pos, y_pos,
                        x_pos + self.tile_size, y_pos + self.tile_size,
                        fill="lime green", tags="snake_body"
                    )

    def _move_snake(self):
        if self.game_over or not self.game_started:
            return

        head_x, head_y = self.snake[0]
        dx, dy = self.snake_direction
        new_head_x = head_x + dx
        new_head_y = head_y + dy

        new_head_x %= self.board_size
        new_head_y %= self.board_size

        new_head = (new_head_x, new_head_y)

        if new_head in self.snake:
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.score_label.configure(text=f"Punkty: {self.score}")
            self.food = self._generate_food()
        else:
            self.snake.pop()

    def update_direction(self, letter):
        if self.game_over or not self.game_started:
            return

        current_dx, current_dy = self.snake_direction
        if letter == "W" and current_dy != 1:
            self.snake_direction = (0, -1)
        elif letter == "S" and current_dy != -1:
            self.snake_direction = (0, 1)
        elif letter == "A" and current_dx != 1:
            self.snake_direction = (-1, 0)
        elif letter == "D" and current_dx != -1:
            self.snake_direction = (1, 0)
        elif letter == "O":
            self.on_closing()

    def game_loop(self):
        if self.game_over:
            self.display_game_over()
            return
        if not self.game_started:
            return

        self._move_snake()
        self._draw_board()
        self.game_loop_id = self.master.after(300, self.game_loop)

    def start_game_loop(self):
        if self.game_loop_id:
            self.master.after_cancel(self.game_loop_id)
        self.game_loop_id = self.master.after(300, self.game_loop)

    def start_countdown(self):
        if self.countdown_val > 0:
            self.canvas.delete("countdown")
            self.canvas.create_text(
                self.board_size * self.tile_size / 2, self.board_size * self.tile_size / 2,
                text=str(self.countdown_val), fill="white", font=("Arial", 80, "bold"), tags="countdown"
            )
            self.countdown_val -= 1
            self.countdown_id = self.master.after(1000, self.start_countdown)
        else:
            self.canvas.delete("countdown")
            self.game_started = True
            self.start_game_loop()

    def display_game_over(self):
        self.canvas.create_text(
            self.board_size * self.tile_size / 2, self.board_size * self.tile_size / 2 - 20,
            text="GAME OVER", fill="white", font=("Arial", 30, "bold"), tags="game_over_text"
        )
        self.canvas.create_text(
            self.board_size * self.tile_size / 2, self.board_size * self.tile_size / 2 + 20,
            text=f"Wynik: {self.score}", fill="white", font=("Arial", 24), tags="game_over_score"
        )
        if self.game_loop_id:
            self.master.after_cancel(self.game_loop_id)

    def on_closing(self):
        global snake_game_running, snake_game_window, snake_game_controller
        if self.game_loop_id:
            self.master.after_cancel(self.game_loop_id)
        if hasattr(self, 'countdown_id') and self.countdown_id is not None:
            self.master.after_cancel(self.countdown_id)

        snake_game_running = False
        snake_game_window = None
        snake_game_controller = None
        self.master.destroy()


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
    if event.keysym == "s" and not snake_game_running:
        launch_snake_game()


app.bind("<Key>", key_event)
app.after(0, update_frame)
app.mainloop()

cap.release()
cv2.destroyAllWindows()
