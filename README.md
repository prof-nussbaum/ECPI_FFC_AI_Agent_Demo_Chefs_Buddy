# Chef's Buddy – Autonomous AI Agent Demo (Fall Faculty Conference)

This repo contains a live, visual demo of **autonomous AI agents** for restaurant/culinary use cases (“Chef’s Buddy”). It was originally built for a Fall Faculty Conference presentation and includes:

- A full-screen **PyGame slideshow** with forward/back navigation
- Pre-generated slides plus **LLM-driven slides**
- A set of **LangGraph workflows** that:
  - Answer natural-language questions with context (`generic_graph`)
  - Turn recipes into **image prompts + Stable Diffusion images**
  - Generate **Python code** from CSVs, execute it, and visualize results
- Support for:
  - **Sentiment analysis** of customer reviews
  - **Recipe ideation + prep lists**
  - **Wine pairing**
  - **Spreadsheet Q&A and visualization**
  - **Image-to-table** conversion with a VLM

---

## Requirements

- Python 3.10+ (3.11 / 3.12 usually ok)
- [Ollama](https://ollama.com/) running locally with models:
  - `llama3.2:3b`
  - `llama3.2:1b`
  - `gemma3:12b-it-qat`
  - `codellama`
  - `llava` (or `gemma3:12b-it-qat` for the advanced VLM demo)
- A running **Stable Diffusion WebUI** API server  
  (e.g. AUTOMATIC1111 on `http://127.0.0.1:7860/`)

### Python packages

Install with `pip`:

```bash
pip install \
  langchain-ollama \
  langchain-community \
  langgraph \
  transformers \
  pygame \
  selenium \
  webdriver-manager \
  pandas \
  matplotlib \
  pillow \
  requests \
  ipython

---
You also need a tokenizer.json file in the project root, used only for counting tokens and characters. You can reuse a Hugging Face tokenizer JSON.

Data files and relative paths

All data paths are relative to the project root (where you run python main.py):

Slideshow images: ./data/FFC_Slides/slide_pics/Slide1.jpg, Slide2.jpg, …

CSVs & text:

./data/orders.csv

./data/Culinary Spreadsheet 2 (Inventory).csv

./data/Wine_list.txt

./data/Salmon recipe.txt

HTML:

./data/customer_reviews.html

./data/business_recommendation.html (overwritten by the demo)

IDEAS_AND_RECIPES.html (written in project root)

Invoice image:

./data/INVOICE-zoom.png

Generated scratch images:

./data/left_page.png

./data/right_page.png

Various idea plating PNGs (e.g. Idea_1_Plating.png)

Recipe HTML pages (e.g. Grilled Salmon with White Rice...html)

These paths and filenames are preserved exactly.

How to run

From the project root:

python main.py


Controls:

Right Arrow → next slide

Left Arrow → previous slide

Some slides trigger live LLM / VLM / Stable Diffusion demos.

demo mode can auto-advance slides (set in AI_Agent_Demo_v01.py).

To exit, simply close the PyGame window.

Code structure

main.py
Thin entrypoint that calls run_demo().

AI_Agent_Demo_v01.py
Core demo logic:

Model configuration (Ollama LLMs, embeddings, tokenizer)

LangGraph State and flows:

generic_graph

generate_image_graph

generate_plot_graph

Demo functions:

Sentiment analysis → business_recommendation.html

Recipe to captioned image

Wine pairing

CSV Q&A, CSV→HTML

Image-to-CSV using VLM

Recipe ideas + prep + wine pairing loop

Slideshow wiring and event loop

FFC_utils.py
Display & slideshow helpers:

PyGame window & layout

SlideShow class

setup_screen(), update_display()

html_to_png() using headless Chrome + Selenium

close_down_show()

FFC_Functions.py
Small self-contained example:

LLM + embeddings + tokenizer

pandas_codewriter() asks the LLM to generate a Pandas snippet over Inventory.csv


