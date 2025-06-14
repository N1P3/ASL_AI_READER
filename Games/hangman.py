import customtkinter as ctk
import random
import os
from PIL import Image, ImageTk


class HangmanGame:
    def __init__(self, master, get_current_letter_callback, scale=2):
        self.master = master
        self.get_current_letter_callback = get_current_letter_callback
        self.scale = scale
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master.geometry("800x800")

        self.word_list = self._load_words()
        self.word_to_guess = random.choice(self.word_list).upper()
        self.guessed_letters = set()
        self.remaining_lives = 6

        self.word_label = ctk.CTkLabel(self.master, text=self._get_display_word(), font=("Arial", 24, "bold"))
        self.word_label.pack(pady=10)

        self.info_label = ctk.CTkLabel(self.master, text=f"Pozostałe próby: {self.remaining_lives}", font=("Arial", 16))
        self.info_label.pack(pady=5)

        self.guessed_label = ctk.CTkLabel(self.master, text="Odgadnięte litery: ", font=("Arial", 16))
        self.guessed_label.pack(pady=5)

        self.canvas = ctk.CTkCanvas(self.master, width=200 * self.scale, height=300 * self.scale, bg="white", highlightthickness=0)
        self.canvas.pack(pady=10)

        self.textures = {}
        parts = ["head", "body", "left_arm", "right_arm", "left_leg", "right_leg"]
        for part in parts:
            path = os.path.join("textures", "hangman", f"{part}.png")
            try:
                size = (40 * self.scale, 40 * self.scale)
                image = Image.open(path).resize(size, Image.LANCZOS)
                self.textures[part] = ImageTk.PhotoImage(image)
            except Exception as e:
                print(f"Nie można wczytać tekstury {part}:", e)
                self.textures[part] = None

        self.draw_hangman()
        self.master.bind("<Return>", self.on_enter_key)

    def _load_words(self):
        with open("Games/words.txt", "r", encoding="utf-8") as f:
            words = f.read().splitlines()
        return [word.strip() for word in words if word.strip()]

    def _get_display_word(self):
        return " ".join([letter if letter in self.guessed_letters else "_" for letter in self.word_to_guess])

    def update_ui(self):
        self.word_label.configure(text=self._get_display_word())
        self.info_label.configure(text=f"Pozostałe próby: {self.remaining_lives}")
        self.guessed_label.configure(text=f"Odgadnięte litery: {', '.join(sorted(self.guessed_letters))}")
        self.draw_hangman()

    def draw_hangman(self):
        s = self.scale
        self.canvas.delete("all")
        self.canvas.create_line(20 * s, 280 * s, 180 * s, 280 * s, width=4 * s)
        self.canvas.create_line(50 * s, 280 * s, 50 * s, 50 * s, width=4 * s)
        self.canvas.create_line(50 * s, 50 * s, 130 * s, 50 * s, width=4 * s)
        self.canvas.create_line(130 * s, 50 * s, 130 * s, 80 * s, width=4 * s)

        parts_to_draw = 6 - self.remaining_lives

        base_x = 110 * s
        base_y = 120 * s
        positions = {
            "head": (base_x, base_y - 40 * s),
            "body": (base_x, base_y),
            "left_arm": (base_x - 40 * s, base_y),
            "right_arm": (base_x + 40 * s, base_y),
            "left_leg": (base_x - 20 * s, base_y + 40 * s),
            "right_leg": (base_x + 20 * s, base_y + 40 * s),
        }

        if parts_to_draw >= 1 and self.textures.get("head"):
            self.canvas.create_image(*positions["head"], image=self.textures["head"], anchor="nw")
        if parts_to_draw >= 2 and self.textures.get("body"):
            self.canvas.create_image(*positions["body"], image=self.textures["body"], anchor="nw")
        if parts_to_draw >= 3 and self.textures.get("left_arm"):
            self.canvas.create_image(*positions["left_arm"], image=self.textures["left_arm"], anchor="nw")
        if parts_to_draw >= 4 and self.textures.get("right_arm"):
            self.canvas.create_image(*positions["right_arm"], image=self.textures["right_arm"], anchor="nw")
        if parts_to_draw >= 5 and self.textures.get("left_leg"):
            self.canvas.create_image(*positions["left_leg"], image=self.textures["left_leg"], anchor="nw")
        if parts_to_draw >= 6 and self.textures.get("right_leg"):
            self.canvas.create_image(*positions["right_leg"], image=self.textures["right_leg"], anchor="nw")

    def on_enter_key(self, event=None):
        current_letter = self.get_current_letter_callback().upper()
        if not current_letter.isalpha() or len(current_letter) != 1 or current_letter in self.guessed_letters:
            return
        self.guessed_letters.add(current_letter)
        if current_letter not in self.word_to_guess:
            self.remaining_lives -= 1
        self.update_ui()
        if "_" not in self._get_display_word():
            self.end_game(True)
        elif self.remaining_lives <= 0:
            self.end_game(False)

    def end_game(self, won):
        result_text = "WYGRAŁEŚ!" if won else f"PRZEGRAŁEŚ! Słowo: {self.word_to_guess}"
        self.info_label.configure(text=result_text)
        self.master.unbind("<Return>")

    def on_closing(self):
        global hangman_game_running, hangman_game_window, hangman_game_controller
        hangman_game_running = False
        hangman_game_window = None
        hangman_game_controller = None
        self.master.destroy()


if __name__ == "__main__":
    def get_letter():
        return entry.get()

    root = ctk.CTk()
    root.geometry("800x800")
    entry = ctk.CTkEntry(root)
    entry.pack()
    game = HangmanGame(root, get_letter)
    root.mainloop()
