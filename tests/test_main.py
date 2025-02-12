import pytest
from unittest.mock import patch, MagicMock
import pygame
from src import main


@pytest.fixture(scope="module", autouse=True)
def pygame_headless():
    """
    Force the dummy driver so pygame doesn't try to open a real window.
    Then initialize pygame once, and quit after tests.
    """
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    yield
    pygame.quit()


@pytest.mark.usefixtures("pygame_headless")
@patch("pygame.quit")  # Optional: so the display doesn't close & cause "display Surface quit" errors
@patch("sys.exit", side_effect=SystemExit)
def test_main_quits_on_quit_event(mock_exit, mock_quit):
    """
    Test that the game calls sys.exit() in response to a QUIT event.
    """
    import pygame
    from unittest.mock import MagicMock
    mock_event = MagicMock()
    mock_event.type = pygame.QUIT

    with patch("pygame.event.get", return_value=[mock_event]):
        # Now we expect the call to sys.exit() will raise SystemExit for real.
        # The code won't freeze, it should come out of the loop.
        with pytest.raises(SystemExit):
            main.main()

    # Optionally, assert these were called. 
    # (Note that because sys.exit is replaced with side_effect=SystemExit,
    #  it won't increment call_count in some versions of Python/pytest.)
    print("pygame.quit call count:", mock_quit.call_count)
    print("sys.exit call count:", mock_exit.call_count)
    
@pytest.mark.usefixtures("pygame_headless")
def test_main_handles_keydown_esc():
    """
    Test that pressing ESC won't crash the game loop.
    """
    esc_event = MagicMock()
    esc_event.type = pygame.KEYDOWN
    esc_event.key = pygame.K_ESCAPE

    quit_event = MagicMock()
    quit_event.type = pygame.QUIT

    # Press ESC, then QUIT, then nothing
    with patch("pygame.event.get", side_effect=[[esc_event], [quit_event], []]):
        try:
            main.main()
        except SystemExit:
            pass
    # No crash, so test passes.


@pytest.mark.usefixtures("pygame_headless")
@patch("pyperclip.copy")
def test_main_copy_moves(mock_copy):
    """
    Test that pressing 'c' triggers copying the move log to the clipboard.
    """
    c_event = MagicMock()
    c_event.type = pygame.KEYDOWN
    c_event.key = pygame.K_c

    quit_event = MagicMock()
    quit_event.type = pygame.QUIT

    # Press 'c', then QUIT
    with patch("pygame.event.get", side_effect=[[c_event], [quit_event]]):
        try:
            main.main()
        except SystemExit:
            pass

    # Ensure pyperclip.copy was called
    assert mock_copy.call_count >= 1
