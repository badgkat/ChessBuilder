# clock.py
import pygame

def format_time(seconds_left: float) -> str:
    """Turns a float number of seconds into MM:SS format."""
    if seconds_left < 0:
        return "0:00"
    mins = int(seconds_left // 60)
    secs = int(seconds_left % 60)
    return f"{mins}:{secs:02}"

class ChessClock:
    def __init__(self, starting_seconds_white, starting_seconds_black, increment=0):
        """Initialize the clock with starting times in seconds for each side,
        and an increment (in seconds) added after each move (e.g. 2 for 3|2)."""
        self.white_time = float(starting_seconds_white)
        self.black_time = float(starting_seconds_black)
        self.increment = increment
        self.current = 'white'  # whose clock is currently ticking
        self.last_update_time = None  # store last tick in pygame time

    def start(self, color: str):
        """Start or switch to the given color's clock."""
        self.current = color
        self.last_update_time = pygame.time.get_ticks()  # in ms
    
    def stop(self):
        """Stop updating times (optional if we want to freeze)."""
        self.last_update_time = None

    def switch_turn(self):
        """Called when a player finishes a move: apply increment and swap clocks."""
        if self.current == 'white':
            self.white_time += self.increment
            self.current = 'black'
        else:
            self.black_time += self.increment
            self.current = 'white'
        self.last_update_time = pygame.time.get_ticks()

    def update(self):
        """Subtract elapsed time from the currently ticking player's clock."""
        if self.last_update_time is None:
            return
        now = pygame.time.get_ticks()
        dt = (now - self.last_update_time) / 1000.0  # convert ms to seconds
        self.last_update_time = now

        if self.current == 'white':
            self.white_time -= dt
        else:
            self.black_time -= dt

        # You could check if self.white_time <= 0 or self.black_time <= 0 here
        # and declare the game over, etc.
