import pygame
import subprocess
import sys

pygame.init()

WIDTH, HEIGHT = 400, 350  # Increased height for warning message
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Python AI Launcher")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BRUNSWICK_GREEN = (52, 78, 65)
FERN_GREEN = (104, 152, 103)
TIMBERWOLF = (218, 215, 205)
HUNTER_GREEN = (58, 90, 64)
RED = (195, 34, 40)

font1 = pygame.font.Font('./fonts/Death Note.ttf', 29)  # Main button font
font2 = pygame.font.Font('./fonts/SnakeChan-MMoJ.ttf', 20)  # Title font
warning_font = pygame.font.Font('./fonts/SnakeChan-MMoJ.ttf', 12)  # Warning message font

title = font2.render("Python AI Launcher", True, TIMBERWOLF)

buttons = [
    {"rect": pygame.Rect(100, 50, 200, 50), "text": "Original Mode", "script": "org/original_agent_dql.py"},
    {"rect": pygame.Rect(100, 120, 200, 50), "text": "Walls Mode", "script": "walls/walls_agent_dql.py"},
    {"rect": pygame.Rect(100, 190, 200, 50), "text": "Three Apples Mode", "script": "apple/apple_dql.py"}
]

current_process = None  # This will store the currently running process
warning_text = ""  # Store warning message

running = True
while running:
    screen.fill(BRUNSWICK_GREEN)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            for button in buttons:
                if button["rect"].collidepoint(mouse_pos):
                    # Only launch if no process is running or if the previous process has finished
                    if current_process is None or current_process.poll() is not None:
                        current_process = subprocess.Popen([sys.executable, button["script"]])
                        warning_text = ""  # Clear warning when new process starts
                    else:
                        warning_text = "A process is already running. Please wait!"

    # Draw title
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 10))

    # Draw buttons
    for button in buttons:
        pygame.draw.rect(screen, FERN_GREEN, button["rect"])
        text_surface = font1.render(button["text"], True, TIMBERWOLF)
        text_rect = text_surface.get_rect(center=button["rect"].center)
        screen.blit(text_surface, text_rect)

    # Display warning text below the last button
    if warning_text:
        warning_surface = warning_font.render(warning_text, True, RED)
        screen.blit(warning_surface, (WIDTH // 2 - warning_surface.get_width() // 2, 270))

    pygame.display.flip()

pygame.quit()
