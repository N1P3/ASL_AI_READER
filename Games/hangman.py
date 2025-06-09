import customtkinter as ctk
import random

class HangmanGame:
    def __init__(self, master, get_current_letter_callback):
        self.master = master
        self.get_current_letter_callback = get_current_letter_callback
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

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

        self.canvas = ctk.CTkCanvas(self.master, width=200, height=300, bg="white", highlightthickness=0)
        self.canvas.pack(pady=10)

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
        self.canvas.delete("all")
        self.canvas.create_line(20, 280, 180, 280, width=4)
        self.canvas.create_line(50, 280, 50, 50, width=4)
        self.canvas.create_line(50, 50, 130, 50, width=4)
        self.canvas.create_line(130, 50, 130, 80, width=4)

        parts_to_draw = 6 - self.remaining_lives

        if parts_to_draw >= 1:
            self.canvas.create_oval(110, 80, 150, 120, width=4)
        if parts_to_draw >= 2:
            self.canvas.create_line(130, 120, 130, 200, width=4)
        if parts_to_draw >= 3:
            self.canvas.create_line(130, 140, 100, 170, width=4)
        if parts_to_draw >= 4:
            self.canvas.create_line(130, 140, 160, 170, width=4)
        if parts_to_draw >= 5:
            self.canvas.create_line(130, 200, 110, 250, width=4)
        if parts_to_draw >= 6:
            self.canvas.create_line(130, 200, 150, 250, width=4)

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
