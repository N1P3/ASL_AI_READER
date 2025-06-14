import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np

ctk.set_appearance_mode("dark")


class SnakeGame:
    def __init__(self, master):
        self.master = master
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.score = 0
        self.game_over = False
        self.game_started = False
        self.board_size = 10
        self.tile_size = 80
        self.snake = [(5, 5)]
        self.snake_direction = (0, 1)
        self.food = self._generate_food()
        self._load_textures()
        self.canvas = ctk.CTkCanvas(
            self.master,
            width=self.board_size * self.tile_size,
            height=self.board_size * self.tile_size,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(pady=10)
        self.score_label = ctk.CTkLabel(
            self.master, text=f"Punkty: {self.score}", font=("Arial", 24, "bold")
        )
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
            "body_straight": "textures/body_vertical.png",
            "corner_ld": "textures/body_corner_ld.png",
            "corner_ru": "textures/body_corner_ru.png",
            "tail_base": "textures/tail_up.png",
            "food": "textures/food.png",
        }

        def load(path):
            try:
                return Image.open(path).resize(
                    (self.tile_size, self.tile_size), Image.Resampling.LANCZOS
                )
            except Exception:
                return Image.new(
                    "RGBA", (self.tile_size, self.tile_size), (120, 120, 120, 255)
                )

        self.head_photo_images = {
            (0, -1): ImageTk.PhotoImage(load(texture_paths["head_up"])),
            (0, 1): ImageTk.PhotoImage(load(texture_paths["head_down"])),
            (-1, 0): ImageTk.PhotoImage(load(texture_paths["head_left"])),
            (1, 0): ImageTk.PhotoImage(load(texture_paths["head_right"])),
        }

        base_straight = load(texture_paths["body_straight"])
        base_tail = load(texture_paths["tail_base"])

        self.body_straight_images = {}
        self.tail_photo_images = {}
        dirlist = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        rot = [0, 180, -90, 90]
        for d, a in zip(dirlist, rot):
            self.body_straight_images[d] = ImageTk.PhotoImage(base_straight.rotate(a))
            self.tail_photo_images[d] = ImageTk.PhotoImage(base_tail.rotate(a))

        base_ld = load(texture_paths["corner_ld"])
        base_ru = load(texture_paths["corner_ru"])

        self.body_corner_images = {
            frozenset({(-1, 0), (0, 1)}): ImageTk.PhotoImage(base_ld),
            frozenset({(1, 0), (0, 1)}): ImageTk.PhotoImage(base_ld.rotate(90)),
            frozenset({(1, 0), (0, -1)}): ImageTk.PhotoImage(base_ld.rotate(180)),
            frozenset({(-1, 0), (0, -1)}): ImageTk.PhotoImage(base_ld.rotate(270)),
            frozenset({(-1, 0), (0, -1)}): ImageTk.PhotoImage(base_ru),
            frozenset({(-1, 0), (0, 1)}): ImageTk.PhotoImage(base_ru.rotate(90)),
            frozenset({(1, 0), (0, 1)}): ImageTk.PhotoImage(base_ru.rotate(180)),
            frozenset({(1, 0), (0, -1)}): ImageTk.PhotoImage(base_ru.rotate(270)),
        }

        self.food_photo_image = ImageTk.PhotoImage(load(texture_paths["food"]))

    def _generate_food(self):
        while True:
            pos = (
                np.random.randint(0, self.board_size),
                np.random.randint(0, self.board_size),
            )
            if pos not in self.snake:
                return pos

    @staticmethod
    def _get_segment_direction(current, other):
        dx = current[0] - other[0]
        dy = current[1] - other[1]
        return (int(np.sign(dx)) if dx else 0, int(np.sign(dy)) if dy else 0)

    def _draw_board(self):
        self.canvas.delete("all")
        for x in range(self.board_size):
            for y in range(self.board_size):
                col = "gray10" if (x + y) % 2 == 0 else "gray15"
                self.canvas.create_rectangle(
                    x * self.tile_size,
                    y * self.tile_size,
                    (x + 1) * self.tile_size,
                    (y + 1) * self.tile_size,
                    fill=col,
                    outline="",
                )

        self.canvas.create_image(
            self.food[0] * self.tile_size + self.tile_size // 2,
            self.food[1] * self.tile_size + self.tile_size // 2,
            image=self.food_photo_image,
            tags="food",
        )

        for i, seg in enumerate(self.snake):
            cx = seg[0] * self.tile_size + self.tile_size // 2
            cy = seg[1] * self.tile_size + self.tile_size // 2
            if i == 0:
                self.canvas.create_image(
                    cx, cy, image=self.head_photo_images[self.snake_direction], tags="snake"
                )
            elif i == len(self.snake) - 1:
                prev = self.snake[i - 1]
                d = self._get_segment_direction(prev, seg)
                self.canvas.create_image(cx, cy, image=self.tail_photo_images[d], tags="snake")
            else:
                prev = self.snake[i - 1]
                nxt = self.snake[i + 1]
                d_prev = self._get_segment_direction(seg, prev)
                d_next = self._get_segment_direction(seg, nxt)
                if d_prev[0] == 0 and d_next[0] == 0:
                    img = self.body_straight_images[(0, 1)]
                elif d_prev[1] == 0 and d_next[1] == 0:
                    img = self.body_straight_images[(1, 0)]
                else:
                    base_set = frozenset({d_prev, d_next})
                    img = self.body_corner_images[base_set]
                self.canvas.create_image(cx, cy, image=img, tags="snake")

    def _move_snake(self):
        if self.game_over or not self.game_started:
            return
        head_x, head_y = self.snake[0]
        dx, dy = self.snake_direction
        new_head = ((head_x + dx) % self.board_size, (head_y + dy) % self.board_size)
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
        dx, dy = self.snake_direction
        if letter == "W" and dy != 1:
            self.snake_direction = (0, -1)
        elif letter == "S" and dy != -1:
            self.snake_direction = (0, 1)
        elif letter == "A" and dx != 1:
            self.snake_direction = (-1, 0)
        elif letter == "D" and dx != -1:
            self.snake_direction = (1, 0)

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
                self.board_size * self.tile_size / 2,
                self.board_size * self.tile_size / 2,
                text=str(self.countdown_val),
                fill="white",
                font=("Arial", 80, "bold"),
                tags="countdown",
            )
            self.countdown_val -= 1
            self.countdown_id = self.master.after(1000, self.start_countdown)
        else:
            self.canvas.delete("countdown")
            self.game_started = True
            self.start_game_loop()

    def display_game_over(self):
        self.canvas.create_text(
            self.board_size * self.tile_size / 2,
            self.board_size * self.tile_size / 2 - 20,
            text="GAME OVER",
            fill="white",
            font=("Arial", 30, "bold"),
            tags="game_over_text",
        )
        self.canvas.create_text(
            self.board_size * self.tile_size / 2,
            self.board_size * self.tile_size / 2 + 20,
            text=f"Wynik: {self.score}",
            fill="white",
            font=("Arial", 24),
            tags="game_over_score",
        )
        if self.game_loop_id:
            self.master.after_cancel(self.game_loop_id)

    def on_closing(self):
        if self.game_loop_id:
            self.master.after_cancel(self.game_loop_id)
        if hasattr(self, "countdown_id") and self.countdown_id is not None:
            self.master.after_cancel(self.countdown_id)
        self.master.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    root.title("Snake")
    game = SnakeGame(root)

    def key_handler(event):
        key = event.char.upper()
        if key in ("W", "A", "S", "D"):
            game.update_direction(key)

    root.bind("<Key>", key_handler)
    root.mainloop()
