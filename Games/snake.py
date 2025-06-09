import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

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

        self.head_photo_images[(0, -1)] = ImageTk.PhotoImage(pil_head_up)
        self.head_photo_images[(0, 1)] = ImageTk.PhotoImage(pil_head_down)
        self.head_photo_images[(-1, 0)] = ImageTk.PhotoImage(pil_head_left)
        self.head_photo_images[(1, 0)] = ImageTk.PhotoImage(pil_head_right)

        base_body_pil = load_image_pil(texture_paths["body_base"], self.tile_size)
        base_tail_pil = load_image_pil(texture_paths["tail_base"], self.tile_size)

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        angles = [0, 180, -90, 90]

        for direction, angle in zip(directions, angles):
            rotated_body_pil = base_body_pil.rotate(angle)
            self.body_photo_images_oriented[direction] = ImageTk.PhotoImage(rotated_body_pil)

            rotated_tail_pil = base_tail_pil.rotate(angle)
            self.tail_photo_images_oriented[direction] = ImageTk.PhotoImage(rotated_tail_pil)

        pil_food = load_image_pil(texture_paths["food"], self.tile_size)
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

        self.canvas.create_image(
            self.food[0] * self.tile_size + self.tile_size // 2,
            self.food[1] * self.tile_size + self.tile_size // 2,
            image=self.food_photo_image, tags="food"
        )

        for i, segment in enumerate(self.snake):
            x_pos = segment[0] * self.tile_size
            y_pos = segment[1] * self.tile_size

            if i == 0:
                head_photo_image = self.head_photo_images.get(self.snake_direction)
                self.canvas.create_image(
                    x_pos + self.tile_size // 2,
                    y_pos + self.tile_size // 2,
                    image=head_photo_image, tags="snake_head"
                )
            elif i == len(self.snake) - 1:
                prev_segment = self.snake[i - 1]
                tail_direction = self._get_segment_direction(segment, prev_segment)
                tail_photo_image = self.tail_photo_images_oriented.get(tail_direction)
                self.canvas.create_image(
                    x_pos + self.tile_size // 2,
                    y_pos + self.tile_size // 2,
                    image=tail_photo_image, tags="snake_tail"
                )
            else:
                prev_segment = self.snake[i - 1]
                body_direction = self._get_segment_direction(segment, prev_segment)
                body_photo_image = self.body_photo_images_oriented.get(body_direction)
                self.canvas.create_image(
                    x_pos + self.tile_size // 2,
                    y_pos + self.tile_size // 2,
                    image=body_photo_image, tags="snake_body"
                )

    def _move_snake(self):
        if self.game_over or not self.game_started:
            return

        head_x, head_y = self.snake[0]
        dx, dy = self.snake_direction
        new_head_x = (head_x + dx) % self.board_size
        new_head_y = (head_y + dy) % self.board_size

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
        if self.game_loop_id:
            self.master.after_cancel(self.game_loop_id)
        if hasattr(self, 'countdown_id') and self.countdown_id is not None:
            self.master.after_cancel(self.countdown_id)
        self.master.destroy()
