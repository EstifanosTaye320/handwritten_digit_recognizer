import pygame
from PIL import Image
from preprocessing import preprocess_image
from model import load_mnist_model, predict_digit
from tkinter import filedialog 
pygame.font.init()

WIDTH = 800
HEIGHT = 600
BG_COLOR = (220, 220, 220)  
TEXT_COLOR = (120, 20, 70)  
HIGHLIGHT_COLOR = (255, 255, 0)  
BUTTON_COLOR = (10, 150, 170) 
BORDER_COLOR = (100, 100, 100) 
FONT = pygame.font.Font(None, 24)  
FONT_BIG = pygame.font.Font(None, 36)
SCROLL_BAR_WIDTH = 15 
SCROLL_HANDLE_HEIGHT = 30  
WORDS_PER_LINE = 50  

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Digit Recognition (Pygame)")

model = load_mnist_model()

image_display = None
prediction_text = None
prediction_surface = None
canvas_surface = None
canvas_img = None
scroll_y = 0
button_rect = None  
scroll_bar_rect = None  
show_intro = True
intro_page_x = WIDTH
intro_text_surface = None
intro_button_rect = None
intro_button_surface = None
predicted_digit = None
probabilities = None

def load_image(filepath):
    """Loads and resizes the image for display"""
    global image_display, canvas_surface, canvas_img, predicted_digit, probabilities
    try:
        img = Image.open(filepath)
        img = img.resize((280, 280), Image.Resampling.LANCZOS)
        image_display = pygame.image.fromstring(img.tobytes(), img.size, img.mode)

        if canvas_img:
            canvas_surface.fill(BG_COLOR)

        canvas_surface = pygame.Surface((280, 280))
        canvas_surface.blit(image_display, (0, 0))
        canvas_img = canvas_surface

        image_data = preprocess_image(filepath)
        predicted_digit, probabilities = predict_digit(model, image_data)
        update_prediction_text(predicted_digit, probabilities)
    except Exception as e:
        print(f"Error loading image: {e}")

def update_prediction_text(predicted_digit, probabilities):
    """Updates the prediction text"""
    global prediction_text, prediction_surface
    prediction_text = (
        f"Predicted Digit: {predicted_digit}"
        f"\nConfidence Level: {probabilities[0][predicted_digit] * 100:.2f}%"
        "\nProbabilities for each digit:"
    )
    for i in range(10):
        prediction_text += f"\nDigit {i}: {probabilities[0][i]:.4f}"

    prediction_surface = FONT.render(prediction_text, True, TEXT_COLOR)

def draw_intro_page():
    """Draws the introductory page"""
    global intro_page_x, intro_text_surface, intro_button_rect, intro_button_surface

    screen.fill(BG_COLOR)

    intro_text = (
        "Welcome to Handwritten Digit Recognizer!"
        "\nThis application uses a trained neural network to recognize handwritten digits from images."
        "\nClick the 'Start' button to begin."
    )
    words = intro_text.split()
    line_height = FONT.get_height() + 5
    current_y = HEIGHT // 2 - 50
    current_line = []

    for word in words:
        if len(" ".join(current_line) + word) <= WORDS_PER_LINE:
            current_line.append(word)
        else:
            text_surface = FONT.render(" ".join(current_line), True, TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, current_y))
            screen.blit(text_surface, text_rect)
            current_line = [word]
            current_y += line_height

    if current_line:
        text_surface = FONT.render(" ".join(current_line), True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(WIDTH // 2, current_y))
        screen.blit(text_surface, text_rect)

    intro_button_rect = pygame.Rect(WIDTH // 2 - 80, HEIGHT // 2 + 50, 160, 40)
    pygame.draw.rect(screen, BUTTON_COLOR, intro_button_rect, border_radius=5)
    intro_button_surface = FONT.render("Start", True, TEXT_COLOR)
    intro_button_text_rect = intro_button_surface.get_rect(center=intro_button_rect.center)
    screen.blit(intro_button_surface, intro_button_text_rect)

    pygame.display.flip()

def draw_main_ui():
    """Draws the main UI elements"""
    global scroll_y, button_rect, scroll_bar_rect, predicted_digit, probabilities

    screen.fill(BG_COLOR)

    title_text = FONT_BIG.render("Digit Recognition", True, TEXT_COLOR)
    title_rect = title_text.get_rect(center=(WIDTH // 2, 50))
    screen.blit(title_text, title_rect)

    button_rect = pygame.Rect(WIDTH // 2 - 80, 100, 160, 40)
    pygame.draw.rect(screen, BUTTON_COLOR, button_rect, border_radius=5)
    button_text = FONT.render("Open Image", True, TEXT_COLOR)
    button_text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, button_text_rect)

    if image_display:
        display_rect = pygame.Rect(50, 150, 280, 280)
        screen.blit(canvas_img, display_rect.move(0, scroll_y))
        pygame.draw.rect(screen, BORDER_COLOR, display_rect, 2)  # Border

    if prediction_text:
        predicted_digit_surface = FONT.render(f"Predicted Digit: {predicted_digit}", True, TEXT_COLOR)
        confidence_surface = FONT.render(f"Confidence Level: {probabilities[0][predicted_digit] * 100:.2f}%", True, TEXT_COLOR)
        probabilities_surface = FONT.render("Probabilities for each digit:", True, TEXT_COLOR)

        predicted_digit_rect = predicted_digit_surface.get_rect(topleft=(350, 150))
        confidence_rect = confidence_surface.get_rect(topleft=(predicted_digit_rect.left, predicted_digit_rect.bottom + 10))
        probabilities_rect = probabilities_surface.get_rect(topleft=(confidence_rect.left, confidence_rect.bottom + 10))

        screen.blit(predicted_digit_surface, predicted_digit_rect.move(0, scroll_y))
        screen.blit(confidence_surface, confidence_rect.move(0, scroll_y))
        screen.blit(probabilities_surface, probabilities_rect.move(0, scroll_y))

        y_offset = probabilities_rect.bottom + 5  
        for i in range(10):
            digit_text = f"Digit {i}: {probabilities[0][i]:.4f}"
            digit_surface = FONT.render(digit_text, True, TEXT_COLOR)
            digit_rect = digit_surface.get_rect(topleft=(probabilities_rect.left, y_offset))
            screen.blit(digit_surface, digit_rect.move(0, scroll_y))
            y_offset += FONT.get_height() + 5

    scroll_bar_rect = pygame.Rect(WIDTH - SCROLL_BAR_WIDTH, 150, SCROLL_BAR_WIDTH, 280)
    pygame.draw.rect(screen, BORDER_COLOR, scroll_bar_rect)

    scroll_handle_rect = pygame.Rect(scroll_bar_rect.left, scroll_bar_rect.top + scroll_y, SCROLL_BAR_WIDTH, SCROLL_HANDLE_HEIGHT)
    pygame.draw.rect(screen, BUTTON_COLOR, scroll_handle_rect)

    pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if show_intro:
                if intro_button_rect.collidepoint(event.pos):
                    show_intro = False
                    intro_page_x = WIDTH
            else:
                if button_rect.collidepoint(event.pos): 
                    filepath = filedialog.askopenfilename(
                        initialdir="/",
                        title="Select an image",
                        filetypes=(("Image files", "*.jpg *.jpeg *.png *.gif"), ("all files", "*.*"))
                    )
                    if filepath:
                        load_image(filepath)
                elif scroll_bar_rect.collidepoint(event.pos): 
                    scroll_y = event.pos[1] - scroll_bar_rect.top - SCROLL_HANDLE_HEIGHT // 2

    for event in pygame.event.get():
        if event.type == pygame.MOUSEWHEEL:
            scroll_y -= event.y * 30 
            scroll_y = max(0, min(scroll_y, 280 - 280)) 

    if show_intro:
        intro_page_x -= 10
        if intro_page_x < -WIDTH:
            intro_page_x = WIDTH
        screen.fill(BG_COLOR)
        draw_intro_page()
    else:
        draw_main_ui()

pygame.quit()