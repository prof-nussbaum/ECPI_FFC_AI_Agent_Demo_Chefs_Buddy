"""
FFC_utils.py

Utility functions and classes for the Chef's Buddy demo slideshow:

- Global screen geometry & font sizing
- SlideShow class: pre-generated slides + optional LLM functions
- HTML → PNG conversion via headless Chrome (Selenium)
- Top/bottom banner text and token count display via PyGame

"""  

from __future__ import annotations

import os
import time
from typing import Callable, List, Optional

import pygame
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Optional: if you want automatic ChromeDriver resolution
from webdriver_manager.chrome import ChromeDriverManager

# For potential future interactive Pandas demo (not used by default here)
from FFC_Functions import pandas_codewriter

class SlideShow:
    """
    Simple slideshow abstraction:

    - `pregen_slides`: list of pre-generated JPG filepaths
    - `llm_slide_functions`: list of callables for LLM-driven slides
      (must return True if they handled display, False to show pregen image)
    """

    def __init__(self, pregen_slides: List[str], llm_slide_functions: List[Callable[[], bool]]):
        self.pregen_slides = pregen_slides
        self.llm_slide_functions = llm_slide_functions
        self.current_index = -1  # start before first slide

    def _show_slide(self) -> bool:
        """
        Internal helper to show the current slide, calling its associated function
        if any. Falls back to showing the pre-generated slide if the function
        returns False.
        """
        if self.current_index < 0 or self.current_index >= len(self.pregen_slides):
            return False
        fn = self.llm_slide_functions[self.current_index]
        result = fn()
        if result is False:
            show_slide_pic(self.pregen_slides[self.current_index])
        return True

    def forward(self) -> bool:
        """
        Move forward one slide.

        Returns:
            True if a slide was shown, False if we are at the end.
        """
        if self.current_index + 1 >= len(self.pregen_slides):
            print("End of presentation.")
            return False
        self.current_index += 1
        return self._show_slide()

    def backward(self) -> bool:
        """
        Move backward one slide.

        Returns:
            True if a slide was shown, False if we are at the beginning.
        """
        if self.current_index <= 0:
            print("Start of presentation.")
            return False
        self.current_index -= 1
        return self._show_slide()



# Globals shared across the demo
monitor_width: int
monitor_height: int
X: int
Y: int
font_size: int
page_w: int
page_h: int
presentation: SlideShow
driver: webdriver.Chrome

# Data paths
output_path = "./data/"
slide_pics = "./data/FFC_Slides/slide_pics/"


def no_op() -> bool:
    """A null function used when a slide has no LLM content."""
    return False


def setup_screen(llm_functions: List[Callable[[], bool]]) -> None:
    """
    Initialize PyGame window, slideshow object, and a headless Chrome driver.

    Args:
        llm_functions: list of functions to run on specific slides.
                       The mapping from slide index → function is controlled
                       by `llm_indices` below.
    """
    global monitor_width, monitor_height, X, Y, font_size, page_w, page_h, presentation, driver

    pygame.init()
    monitor_width = pygame.display.Info().current_w
    monitor_height = pygame.display.Info().current_h
    X = int(monitor_width * 0.8)
    Y = int(monitor_height * 0.8)
    print(pygame.display.Info().current_w, pygame.display.Info().current_h, X, Y)

    # Page size: the display has a left and right page
    page_h = Y - 80
    page_w = int(X / 2) - 10
    font_size = 36 if monitor_width > 1399 else 24

    # Slideshow wiring: map certain slide indices to LLM functions
    max_steps = 100
    num_steps = 50
    current_step = -1  # unused, but left for parity with original

    pre_made_slides: List[str] = []
    functions: List[Callable[[], bool]] = []
    llm_function_index = 0

    # LLM slides at these indices (1-based slide numbers)
    llm_indices = {7, 9, 13, 15, 20, 24, 26, 31, 35, 36, 41}

    for i in range(num_steps):
        slide_path = os.path.join(slide_pics, f"Slide{i+1}.jpg")
        pre_made_slides.append(slide_path)
        if (i + 1) not in llm_indices:
            functions.append(no_op)
        else:
            functions.append(llm_functions[llm_function_index])
            llm_function_index += 1

    presentation = SlideShow(pre_made_slides, functions)

    # Headless Chrome for HTML → PNG conversion
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument(f"--window-size={page_w - 20},{page_h}")
    chrome_options.add_argument("--hide-scrollbars")

    # Using webdriver_manager for robust driver handling
    driver = webdriver.Chrome(
        options=chrome_options,
        # Uncomment if you want explicit driver version install:
        # service=Service(ChromeDriverManager().install()),
    )
    driver.set_window_size(page_w, page_h * 2)

    print(
        "DONE SETUP - monitor_width, monitor_height, X, Y, font_size, page_w, page_h,",
        monitor_width,
        monitor_height,
        X,
        Y,
        font_size,
        page_w,
        page_h,
        presentation,
    )


def html_to_png(html_text: str, output_path: str = "output.png") -> None:
    """
    Convert an HTML string to a PNG image using headless Chrome.

    Args:
        html_text: HTML markup to render.
        output_path: Where the PNG should be saved.
    """
    global driver

    temp_html = "temp.html"
    try:
        with open(temp_html, "w", encoding="utf-8") as f:
            f.write(html_text)

        driver.get(f"file://{os.path.abspath(temp_html)}")
        time.sleep(1)
        driver.save_screenshot(output_path)

        if os.path.exists(output_path):
            print(f"Successfully saved PNG to {output_path}")
        else:
            print("Failed to create output file")
    finally:
        if os.path.exists(temp_html):
            os.remove(temp_html)


def show_slide_pic(filename: str = "./dummy.jpg") -> None:
    """
    Display a single full-screen slide image using PyGame.
    """
    global X, Y

    screen = pygame.display.set_mode((X, Y + 72))
    screen.fill((0, 0, 0))
    picture = pygame.image.load(filename)
    picture = pygame.transform.scale(picture, (X, Y))
    screen.blit(picture, (0, 0))
    pygame.display.flip()


def update_display(
    left: Optional[str] = None,
    right: Optional[str] = None,
    TX_tokn: int = 0,
    RX_tokn: int = 0,
) -> None:
    """
    Update the main display with left and/or right PNGs and header text.

    Args:
        left: path to left image (or None / False)
        right: path to right image (or None / False)
        TX_tokn: tokens sent to LLM so far (for display)
        RX_tokn: tokens received from LLM so far (for display)
    """
    global X, Y, font_size, page_w

    screen = pygame.display.set_mode([X, Y + 72])
    font = pygame.font.Font(None, font_size)
    screen.fill((0, 0, 0))

    if left:
        left_image = pygame.image.load(left).convert()
        screen.blit(left_image, (0, 72))

    if right:
        right_image = pygame.image.load(right).convert()
        screen.blit(right_image, (page_w + 10, 72))

    heading_text = " --- DEMONSTRATION: Understanding Autonomous AI Agents (Chef's Buddy) --- "
    pygame.display.set_caption(heading_text)
    text_surface = font.render(heading_text, True, (255, 255, 255))
    screen.blit(text_surface, (10, 10))

    sub_head_left = f"Request to Agent: (sent tokens={TX_tokn})"
    text_surface = font.render(sub_head_left, True, (255, 0, 0))
    screen.blit(text_surface, (10, 36))

    sub_head_right = f"Agent's Response: (received tokens={RX_tokn})"
    text_surface = font.render(sub_head_right, True, (0, 255, 0))
    screen.blit(text_surface, (page_w + 20, 36))

    pygame.display.flip()
    print(
        "DONE DISPLAY - monitor_width, monitor_height, X, Y, font_size, page_w, page_h,",
        monitor_width,
        monitor_height,
        X,
        Y,
        font_size,
        page_w,
        page_h,
        presentation,
    )
    time.sleep(3)


def close_down_show() -> None:
    """
    Cleanly shut down the slideshow: close Chrome driver and PyGame.
    """
    global driver
    driver.quit()
    pygame.display.quit()
    pygame.quit()
