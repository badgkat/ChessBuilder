import pytest
import pygame
from clock import ChessClock, format_time

@pytest.fixture
def chess_clock():
    """Fixture to create a ChessClock instance with 5 minutes for both players and a 2-second increment."""
    pygame.time.get_ticks = lambda: 1000  # Mock initial time
    return ChessClock(starting_seconds_white=300, starting_seconds_black=300, increment=2)

def test_format_time():
    """Test the format_time function for correct MM:SS output."""
    assert format_time(0) == "0:00"
    assert format_time(5) == "0:05"
    assert format_time(65) == "1:05"
    assert format_time(600) == "10:00"
    assert format_time(-5) == "0:00"  # Should return "0:00" for negative values

def test_clock_initialization(chess_clock):
    """Ensure the ChessClock initializes with correct values."""
    assert chess_clock.white_time == 300
    assert chess_clock.black_time == 300
    assert chess_clock.increment == 2
    assert chess_clock.current == 'white'
    assert chess_clock.last_update_time is None

def test_start_clock(chess_clock):
    """Ensure the clock starts correctly."""
    pygame.time.get_ticks = lambda: 5000  # Mock time at 5000ms
    chess_clock.start('black')
    assert chess_clock.current == 'black'
    assert chess_clock.last_update_time == 5000

def test_switch_turn(chess_clock):
    """Ensure turn switching works and increments are applied correctly."""
    pygame.time.get_ticks = lambda: 2000  # Mock time at 2000ms
    chess_clock.start('white')
    chess_clock.switch_turn()
    assert chess_clock.current == 'black'
    assert chess_clock.white_time == 302  # Increment applied to white before switching
    assert chess_clock.last_update_time == 2000

    chess_clock.switch_turn()
    assert chess_clock.current == 'white'
    assert chess_clock.black_time == 302  # Increment applied to black before switching

def test_update_time(chess_clock):
    pygame.time.get_ticks = lambda: 1000  # Mock initial time
    chess_clock.start('white')

    pygame.time.get_ticks = lambda: 4000  # Advance time (3 seconds)
    chess_clock.update()
    assert round(chess_clock.white_time, 1) == 297.0  # 300 - 3 = 297

    chess_clock.switch_turn()  # White gets +2s increment
    pygame.time.get_ticks = lambda: 7000  # Advance time (3 more seconds)
    chess_clock.update()
    assert round(chess_clock.black_time, 1) == 297.0  # 300 - 3 = 297

def test_clock_stop(chess_clock):
    """Ensure stopping the clock prevents time updates."""
    pygame.time.get_ticks = lambda: 1000
    chess_clock.start('white')

    pygame.time.get_ticks = lambda: 4000  # 3 seconds pass
    chess_clock.stop()
    chess_clock.update()

    assert round(chess_clock.white_time, 1) == 300.0  # Time should not change
    assert chess_clock.last_update_time is None  # Clock is stopped
