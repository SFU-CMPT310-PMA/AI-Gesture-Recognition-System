import pygame
import sys

pygame.init()
pygame.display.set_caption("Rock-Paper-Scissors Instructions")

WIDTH = 1000
HEIGHT = 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
BEIGE = (245, 222, 179)
WHITE = (255, 255, 255)
DARK = (40, 40, 40)

font_path = "fonts/Press_Start_2P/PressStart2P-Regular.ttf"  # path to font file
FONT_TITLE = pygame.font.Font(font_path, 27)
FONT_TEXT = pygame.font.Font(font_path, 18)
FONT_SMALL = pygame.font.Font(font_path, 15)

INC = 40

"""
   2. TODO Implement a more aesthetic interface.
"""
# Instructions
slides = [
    {
        "title": "Welcome to Rock • Paper • Scissors!",
        "text": [
            "We will first start with collecting",
            "hand gesture data to train your AI opponent!",
            "",
            "Use the → arrow to continue."
        ]
    },
    {
        "title": "Step 1: Get Ready",
        "text": [
            "Make sure your webcam is turned on.",
            "You'll show one hand clearly to the camera.",
            "",
        ]
    },
    {
        "title": "Step 2: Data Collection",
        "text": [
            "Press 'r' to record ROCK",
            "Press 'p' to record PAPER",
            "Press 's' to record SCISSORS",
            "",
            "All samples are auto-saved into a CSV dataset.",
        ]
    },
    # need to implement scan button
    {
        "title": "Step 3: Start the Game",
        "text": [
            "When you're ready, press the SCAN button.",
            "",
            "This will launch the hand tracking window.",
        ]
    }
]

current_slide = 0

def instruction_slide(index):
    SCREEN.fill(BEIGE)
    slide = slides[index]
    
    # to display -> blit, pair with render for text
    # https://stackoverflow.com/questions/20842801/how-to-display-text-in-pygame
    width_text = WIDTH/2
    spacing = 180
    title_header = FONT_TITLE.render(slide["title"], True, DARK)
    SCREEN.blit(title_header, (width_text - title_header.get_width()/2, spacing/4))

    for line in slide["text"]:
        text_surface = FONT_TEXT.render(line, True, DARK)
        SCREEN.blit(text_surface, (width_text - text_surface.get_width()/2, spacing))
        spacing += INC

    how_to_navigate = FONT_SMALL.render("Use ← or → arrows to navigate", True, WHITE)
    SCREEN.blit(how_to_navigate, (width_text - how_to_navigate.get_width()/2, HEIGHT - INC))

    pygame.display.update()

def main():
    global current_slide
    while True:
        instruction_slide(current_slide)
        # end
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # navigate
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and (current_slide < (len(slides) - 1)):
                    current_slide += 1
                if event.key == pygame.K_LEFT and (current_slide > 0):
                    current_slide -= 1

"""
   1. TODO Add a button that starts the data collection process
"""

if __name__ == "__main__":
    main()
