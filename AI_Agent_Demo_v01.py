"""
AI_Agent_Demo_v01.py

Chef's Buddy / Fall Faculty Conference autonomous agent demo.

This module:

- Configures several local LLMs via Ollama (normal, creative, tiny, smart, code).
- Sets up a generic LangGraph pipeline that:
  - Renders "prompt + context" on the left side of the screen,
  - Sends requests to the appropriate LLM,
  - Renders responses on the right side.
- Defines specialized graphs for:
  - Text → image description → Stable Diffusion (`generate_image_graph`)
  - CSV → Python code → image (`generate_plot_graph`)
- Implements individual demo functions:
  - Sentiment analysis of reviews
  - Recipe to captioned image
  - Wine pairing
  - Spreadsheet analysis and visualization
  - Image to CSV using a VLM
  - Recipe ideation with customer feedback loop
- Wires everything into a PyGame slideshow driven by left/right arrow keys.

"""

from __future__ import annotations

import base64
import time
from typing import Literal, Tuple

import numpy
import pandas as pd
import pygame
import requests
from IPython.display import Image
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langgraph.graph import StateGraph
from PIL import Image as PILImage
from transformers import PreTrainedTokenizerFast
from typing_extensions import TypedDict

from FFC_utils import (
    SlideShow,
    close_down_show,
    html_to_png,
    setup_screen,
    update_display,
    output_path,
)

#############################
# Demo flags
#############################

demo: bool = False        # True → auto-advance slides
demo_delay: int = 3       # seconds before auto-pressing right arrow in demo mode

#############################
# Global TX/RX counters
#############################

TX_char = 0
TX_tokn = 0
RX_char = 0
RX_tokn = 0

#############################
# LLM / embeddings configuration
#############################

print(
    """
#############################
#### SETUP INFERENCE AND ENVIRONMENT HYPERPARAMETERS
#############################
"""
)

print(
    f"# running count of token TX/RX {TX_tokn}/{RX_tokn} and characters "
    f"TX to LLM, RX received back from LLM {TX_char}/{RX_char}"
)

local_llm = "llama3.2:3b"          # Pretty fast. Good answers.
tiny_local_llm = "llama3.2:1b"     # Tiny, for quick responses.
smart_local_llm = "gemma3:12b-it-qat"
coding_local_llm = "codellama"

num_ctx = 4096  # best seems to be 4096
print(f"# Context token count maximum (contexts+questions can get truncated if longer {num_ctx})")
print(f"# Model Selection - Ollama serving {local_llm}")

print(
    """
#############################
#### SETUP LLM(s)
#############################
"""
)

# Main LLMs
llm = ChatOllama(model=local_llm, temperature=0, num_ctx=num_ctx)
creative_llm = ChatOllama(model=local_llm, temperature=5.0, num_ctx=num_ctx)
tiny_llm = ChatOllama(model=tiny_local_llm, temperature=0, num_ctx=num_ctx)
smart_llm = ChatOllama(model=smart_local_llm, temperature=0, num_ctx=num_ctx)
code_llm = ChatOllama(model=coding_local_llm, temperature=0, num_ctx=num_ctx)
llm_json_mode = ChatOllama(
    model=local_llm,
    temperature=0,
    format="json",
    num_ctx=num_ctx,
)

embeddings = OllamaEmbeddings(model="llama3")

print(
    """
#############################
#### Function to count characters and tokens
#############################
"""
)

llama_tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer.json",
    clean_up_tokenization_spaces=True,
)


def token_len(text: str) -> Tuple[int, int]:
    """
    Count characters and tokens in `text` using the local tokenizer.json.
    """
    tokens = len(llama_tokenizer.encode(text=text))
    characters = len(text)
    return characters, tokens


print(
    """
#############################
#### save_graph_image SAVE IMAGE OF LANGGRAPH GRAPH DIAGRAM
#############################
"""
)

from IPython.display import Image as IPyImage  # rename to avoid collision
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles


def save_graph_image(graph, filename: str) -> None:
    """
    Save a Mermaid-rendered PNG of a LangGraph graph for documentation.
    """
    with open(filename, "wb") as file:
        file.write(IPyImage(graph.get_graph().draw_mermaid_png()).data)


print(
    """
#############################
#### Setup scratch area to store images
#############################
"""
)

# As we convert page text into images, we'll store those images in these files:
left_page = "left_page.png"
right_page = "right_page.png"
left = output_path + left_page
right = output_path + right_page

salmon_page = "Grilled Salmon with White Rice and Steamed Vegetables.html"

print(
    """
#############################
#### Setup Important Prompts
#############################
"""
)

rag_prompt = """
{question}
{context} 
"""

# Final version of the plating prompt (matches original last assignment)
recipe_to_captioned_image = (
    "Describe an appetizing plating presentation of the below recipe that will be "
    "converted into a 512x512 image by a text-to-image diffusion model. Output the "
    "tasty description as an HTML formatted web page with the name of the dish on "
    "top in large font, then the description in medium font, and at the bottom will "
    "be the image which will be stored in location ./data/{image_filename} - Do not "
    "provide any other introductory text or explanations. Just the HTML formatted "
    "plating description and the image below it. Here is the recipe: {context}"
)

print("#############################")
print("#### CREATING STATE VARIABLE FOR GENERIC GRAPH FUNCTION")
print("#############################")
print("Define state variable ")


class State(TypedDict):
    llm: str
    task_name: str
    question: str
    simple_question: str
    context: str
    final_answer: str
    image_filename: str
    image_create_successful: int
    CSV_file: str
    HTML_to_display: str
    recipes: list
    menu_pages: list


print(
    """
#############################
#### Setup Live LLM Demonstration Functions
#############################
"""
)

###########################################################
# Generic LangGraph nodes and graphs
###########################################################

def _select_llm(llm_key: str):
    """
    Selects which LLM to use based on the 'llm' string in State.
    """
    if llm_key == "smart":
        print("!!!!!!!!!!!!!!!!!!!!! SMART LLM !!!!!!!!!!!!!!!!!!")
        return smart_llm
    if llm_key == "code":
        print("!!!!!!!!!!!!!!!!!!!!! CODE LLM !!!!!!!!!!!!!!!!!!")
        return code_llm
    if llm_key == "tiny":
        print("!!!!!!!!!!!!!!!!!!!!! TINY LLM !!!!!!!!!!!!!!!!!!")
        return tiny_llm
    if llm_key == "creative_llm":
        print("!!!!!!!!!!!!!!!!!!!!! CREATIVE LLM !!!!!!!!!!!!!!!!!!")
        return creative_llm
    print("!!!!!!!!!!!!!!!!!!!!! NORMAL LLM !!!!!!!!!!!!!!!!!!")
    return llm


def generic_node(state: State):
    """
    Generic LangGraph node used by many demo flows.

    - Formats rag_prompt(question, context)
    - Trims context if token limit exceeded
    - Shows prompt/context on the left page
    - Invokes selected LLM
    - Shows response on the right page

    Returns:
        {'final_answer': <LLM response>}
    """
    global TX_char, TX_tokn, RX_char, RX_tokn
    global left, right

    task_name = state["task_name"]
    question = state["question"]
    simple_question = state["simple_question"]
    context = state["context"]

    rag_prompt_formatted = rag_prompt.format(context=context, question=question)
    char, tokn = token_len(rag_prompt_formatted)
    warning_message = " "
    if tokn > num_ctx:
        print(f"\n-->WARNING Token count {tokn} exceeded max context token limit of {num_ctx}")
        trim = int((tokn / num_ctx) + 1)
        trim_pct = int((1 - 1 / trim) * 100)
        length = len(context)
        new_length = int(length / trim)
        warning_message = f"===>EXCEEDED TOKEN LENGTH Trimming data by {trim_pct}% !"
        print(warning_message)
        context = context[: new_length - 1]
        rag_prompt_formatted = rag_prompt.format(context=context, question=question)
        char, tokn = token_len(rag_prompt_formatted)
    TX_char += char
    TX_tokn += tokn

    file = state.get("HTML_to_display", "None")
    if file == "None":
        display_context = context
    else:
        with open(output_path + file, "r", encoding="utf-8") as this_file:
            display_context = this_file.read()

    html_to_png(
        "<body><h1>"
        + task_name
        + "</h1><h2>"
        + "Prompt: "
        + simple_question
        + "</h2><p style='color: red;'"
        + warning_message
        + "</p>"
        + display_context,
        output_path=left,
    )
    update_display(left=left, right=None, TX_tokn=TX_tokn, RX_tokn=RX_tokn)

    model = _select_llm(state["llm"])
    generation = model.invoke([HumanMessage(content=rag_prompt_formatted)])
    final_answer = generation.content
    state["final_answer"] = final_answer
    char, tokn = token_len(final_answer)
    RX_char += char
    RX_tokn += tokn
    html_to_png(final_answer, output_path=right)
    update_display(left=left, right=right, TX_tokn=TX_tokn, RX_tokn=RX_tokn)
    print(f"Response is: {final_answer}")
    return {"final_answer": final_answer}


def start_node(state: State):
    """
    A variant of generic_node used when we only need to show
    the initial prompt/context, not the final right-side image update.
    """
    global TX_char, TX_tokn, RX_char, RX_tokn
    global left, right

    task_name = state["task_name"]
    question = state["question"]
    simple_question = state["simple_question"]
    context = state["context"]

    rag_prompt_formatted = rag_prompt.format(context=context, question=question)
    char, tokn = token_len(rag_prompt_formatted)
    if tokn > num_ctx:
        print(f"\n-->WARNING Token count {tokn} exceeded max context token limit of {num_ctx}")
    TX_char += char
    TX_tokn += tokn

    file = state.get("HTML_to_display", "None")
    if file == "None":
        display_context = context
    else:
        with open(output_path + file, "r", encoding="utf-8") as this_file:
            display_context = this_file.read()

    html_to_png(
        "<body><h1>"
        + task_name
        + "</h1><h2>"
        + "Prompt: "
        + simple_question
        + "</h2>"
        + display_context,
        output_path=left,
    )
    update_display(left=left, right=None, TX_tokn=TX_tokn, RX_tokn=RX_tokn)

    model = _select_llm(state["llm"])
    generation = model.invoke([HumanMessage(content=rag_prompt_formatted)])
    final_answer = generation.content
    state["final_answer"] = final_answer
    char, tokn = token_len(final_answer)
    RX_char += char
    RX_tokn += tokn
    html_to_png(final_answer, output_path=right)
    print(f"Response is: {final_answer}")
    return {"final_answer": final_answer}


def generate_image(state: State):
    """
    Node that calls Stable Diffusion WebUI API to generate an image from
    the already-generated image description in state['final_answer'].
    """
    global TX_char, TX_tokn, RX_char, RX_tokn
    global left, right

    image_description = state["final_answer"]
    image_filename = state["image_filename"]
    char, tokn = token_len(image_description)
    if tokn > num_ctx:
        print(f"\n-->WARNING Token count {tokn} exceeded max context token limit of {num_ctx}")
    TX_char += char
    TX_tokn += tokn
    update_display(left=left, right=None, TX_tokn=TX_tokn, RX_tokn=RX_tokn)

    # Stable Diffusion txt2img request
    payload = {
        "prompt": image_description,
        "sampler_name": "Euler",
        "scheduler": "Automatic",
        "steps": 20,
        "width": 512,
        "height": 512,
    }
    response = requests.post(url="http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload)
    r = response.json()
    with open("./data/" + image_filename, "wb") as f:
        f.write(base64.b64decode(r["images"][0]))

    RX_tokn += 1290  # typical image generation token count for output
    print(f"Image Description: {image_description}\n")
    html_to_png(image_description, output_path=right)
    update_display(left=left, right=right, TX_tokn=TX_tokn, RX_tokn=RX_tokn)
    return


def execute_code(state: State):
    """
    Node that executes Python code returned by the code LLM (for CSV visualization).

    - Cleans up the code (strip backticks, remove plt.show, etc.)
    - Prefixes imports and DataFrame loading boilerplate
    - Uses exec() to run the code
    - Shows the code on the left, and the resulting image on the right
    """
    global TX_char, TX_tokn, RX_char, RX_tokn
    global left, right

    task_name = state["task_name"]
    question = state["question"]
    simple_question = state["simple_question"]
    image_filename = state["image_filename"]
    CSV_file = state["CSV_file"]
    python_snippet = str(state["final_answer"])

    python_snippet = python_snippet.replace("python", "")
    python_snippet = python_snippet.replace("plt.show()", "")
    python_snippet = f"{python_snippet}".strip().strip("'").strip("`")
    python_snippet = python_snippet.replace("%matplotlib inline", " ")

    python_preface = (
        "from matplotlib import pyplot as plt\n"
        "import pandas as pd\n"
        f"filepath = './data/{CSV_file}'\n"
        "df = pd.read_csv(filepath, parse_dates=True)\n"
    )

    python_code = python_preface + python_snippet + "\nplt.close()"
    html_python_code = python_code.replace("\n", "<p>")

    html_to_png(
        "<body><h1>"
        + task_name
        + "</h1><h2>Python code:</h2>"
        + html_python_code,
        output_path=left,
    )
    update_display(left=left, right=None, TX_tokn=TX_tokn, RX_tokn=RX_tokn)

    try:
        exec(python_code, {})
        print("Code executed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    else:
        print("No further attempts needed.")

    char, tokn = token_len(python_code)
    RX_char += char
    RX_tokn += tokn

    html_to_png(
        "<body><h2>"
        + simple_question
        + "</h2><img src=./data/"
        + image_filename
        + " style='max-width: 100%; height: auto;'>",
        output_path=right,
    )
    update_display(left=left, right=right, TX_tokn=TX_tokn, RX_tokn=RX_tokn)
    return


#########################
# Graph construction
#########################

# Generic graph
workflow = StateGraph(State)
workflow.add_node("Generic Node", generic_node)
workflow.set_entry_point("Generic Node")
workflow.set_finish_point("Generic Node")
generic_graph = workflow.compile()

# Generate image graph
workflow2 = StateGraph(State)
workflow2.add_node("start_node", start_node)
workflow2.add_node("generate_image", generate_image)
workflow2.set_entry_point("start_node")
workflow2.add_edge("start_node", "generate_image")
workflow2.set_finish_point("generate_image")
generate_image_graph = workflow2.compile()

# Generate plot graph
workflow3 = StateGraph(State)
workflow3.add_node("Generic Node", generic_node)
workflow3.add_node("execute_code", execute_code)
workflow3.set_entry_point("Generic Node")
workflow3.add_edge("Generic Node", "execute_code")
workflow3.set_finish_point("execute_code")
generate_plot_graph = workflow3.compile()

################# END OF LANGGRAPH ############################
###############################################################
####++++++++++++++++++++++++ Recipe Ideas 
def recipe_ideas_function() :
    global TX_char, TX_tokn, RX_char, RX_tokn
    global left, right
    global salmon_page

    max_ideas = 10
    ideas_so_far = ["Grilled Salmon with White Rice and Steamed Vegetables."]
    with open(salmon_page, "r") as file:
        salmon_page_html = file.read()
    recipes = [ideas_so_far]
    html_to_png(salmon_page_html, output_path=right)
    cumulative_menu_pages = " "

    file = "business_recommendation.html"
    with open(output_path+file, 'r', encoding='utf-8') as file:
        business_recommendation = file.read()
########### SUMMARIZE BUSINESS RECOMMENDATIONS
    task_name = "Summarize Business Recommendations"
    question = "Identify the customer recommendations and create a short bullet list in HTML format with large font of those recommendations that came from the following business recommendations from interviewing customers. Do not include introductory nor explanatory text. Only provide the bullet list."
    question = "Create a bullet list (in HTML format) that summarizes the following business recommendations from interviewing customers. Do not include introductory or explanatory text. Only provide the HTML format bullet list."
    context = business_recommendation
    result = generic_graph.invoke({
        "llm": "normal",
        "task_name": task_name,
        "question": question,
        "simple_question": question,
        "context": context,
        "final_answer": " ",
        "HTML_to_display": "None",
    })
    business_recommendation = result["final_answer"]
    print("business_recommendation: ", business_recommendation, "\n")
    
############ LOOP THROUGH THE RECOMMENDATIONS
    for i in range(max_ideas) :
        num_ideas = len(ideas_so_far)
        ideas_html = "<ul>"
        for j in range(num_ideas) :
            ideas_html += "<li>" + ideas_so_far[j] + "</li>"
        ideas_html += "</ul>"
############  COME UP WITH AN IDEA
        task_name = f"Create recipe idea {i+1}"
        prompt_runnable = "I am trying to come up with a new idea on how to prepare salmon along with two or three sides or sauces to accompany that preparation method. Here are the ideas I have so far: {ideas_so_far}. Can you help me come up with one new idea based on a preparation method that is different from the ideas so far? Just provide a single sentence enging with a period '.' with the name of the salmon preparation method followed by two or three sides or sauces, in a similar format to one of the ideas I have so far. Just repond with the single idea sentnece. Do not add any other explanatory text or comments, and do not repeat information from the ideas I have so far."
        prompt_runnable = "I am trying to come up with a new idea on how to prepare salmon along with two or three sides or sauces to accompany that preparation method. Here are the ideas I have so far: {ideas_html}. Can you help me come up with one new idea based on a preparation method that is different from the ideas so far? Here are some suggestions if you need them: {business_recommendation}. Just provide a single sentence enging with a period '.' with the name of the salmon preparation method followed by two or three sides or sauces, in a similar format to one of the ideas I have so far. Just repond with the single idea sentence. Do not add any other explanatory text or comments, and do not repeat information from the ideas I have so far."
        prompt_runnable = "I am trying to come up with a new idea on how to prepare salmon along with two or three sides or sauces to accompany that preparation method. Here are the ideas I have so far: {ideas_so_far}. Can you help me come up with one new idea based on a preparation method that is different from the ideas so far? Just provide a single sentence enging with a period '.' with the name of the salmon preparation method followed by two or three sides or sauces, in a similar format to one of the ideas I have so far. Just repond with the single idea sentnece. Do not add any other explanatory text or comments, and do not repeat information from the ideas I have so far."
        prompt_runnable = "Customers have suggested some changes to the Salmon Main Course as follows: {business_recommendation} Follow the suggestions and choose one of the suggested preparation options along with two or three suggested sides or sauce options. Also, make sure your chosen preparation option, sides, and sauces are different from those I have so far listed here: {ideas_html}. Just provide a single sentence enging with a period '.' with the name of the preparation option followed by two or three sides or sauces, in a similar format to one of the ideas I have so far. Just respond with the new option sentence. Do not add any other explanatory text or comments, and do not repeat information from the ideas I have so far."
 
        question = prompt_runnable.format(
            ideas_so_far = ideas_so_far, 
            ideas_html = ideas_html, 
            business_recommendation = business_recommendation,
        )
        simple_question = f"<h2>Come up with an idea based on customer suggestions, but that is different from one of these ideas:{ideas_html}</h2><h3>You'll recall that the customer suggestions included:</h3>{business_recommendation}"
        result = generic_graph.invoke({
            "llm": "creative_llm", # was "smart", # was "creative_llm", # was "llm",
            "task_name": task_name,
            "question": question,
            "simple_question": simple_question,
            "context": " ",
            "final_answer": " ",
            "image_filename": " ",
            "CSV_file": " ",
            "HTML_to_display": "None",
            "recipes": recipes,
        })
        resp = result["final_answer"]
        print("RESPONSE: ", resp,"\n")
        ideas_so_far.append(resp)
############ MAKE AND SAVE A MARKETING PAGE FOR THE IDEA AND RECIPE AND CREATED GRAPHIC
#        recipe = recipes[i+1]
        recipe = ideas_so_far[i+1]
#        image_filename = ideas_so_far[i+1] + 'png'
        image_filename = f"Idea_{i+1}_Plating.png"
        task_name = f"Convert Recipe {i+1} into a Marketing Page showing a Suggested Plating Presentation"
        context = recipe
        recipe_to_captioned_image_prompt = recipe_to_captioned_image.format(
            context=context, 
            image_filename = image_filename)
        simple_question = "Provide recipe name, appetizing description, and suggested plating image"
        result = generate_image_graph.invoke({
            "llm": "normal",
            "task_name": task_name,
            "question": recipe_to_captioned_image_prompt,
            "simple_question": simple_question, 
            "context": "<h2>" + context + "</h2>",
            "image_filename": image_filename,
            "final_answer": " ",
            "HTML_to_display": "None",
            "recipes": recipes,
        })
        resp = result["final_answer"]
        print("RESPONSE: ", resp,"\n")
        filename = ideas_so_far[i+1] + "html"
        with open(filename, "w") as file:
            file.write(resp)
        cumulative_menu_pages = cumulative_menu_pages + resp 
###### END OF MAKING MENU PAGEFOR IDEA

############## MAKE A PREP COOKING LIST FOR THE IDEA
        prompt_runnable = "Can you please create a recipe for {resp}. Format the recipe into an HTML page. Respond only with the HTML page and no other introductory text or explanatory text. Only reply with the recipe in a format similar to '<body><h1>Name of Recipe</h1><h2>Recipe Ingredients: ...<p>Recipe Procedure: ...</h2>'"
        prompt_runnable = "We will be preparing {resp} and expect to have 5 to 7 orders each evening at our Rustic Vineyard Restaurant. We want a single serving recipe for {resp} and also a HTML formatted bullet list in large font of prep cooking that must be prepared the night before. Can you please create a recipe and prep bullet list for {resp}. Format the recipe and prep into an HTML page. Respond only with the HTML page and no other introductory text or explanatory text. Only reply with the recipe and prep in a format similar to '<body><h1>Name of Recipe</h1><h2>Restaurant Prep: (put bullet list here)</h2><h2>Recipe Ingredients: (put bullet list here)<p>Recipe Procedure Upon Order: (put procedure here)</h2>'"
        prompt_runnable = "We will be preparing {resp} and expect to have 5 to 7 orders each evening at our Rustic Vineyard Restaurant. Please generate an HTML format large font bullet list of prep cooking that must be prepared the night before. Here is an example HTML bullet list format: <h1>Night-Before Prep-Cooking Checklist</h1><ul><li>Prepare vegetables by...</li><li>Prepare sauce by...</li></ul>"
        question = prompt_runnable.format(
            resp = resp, 
        )
        task_name = f"For idea {i+1}, Provide Night-Before Prep-Cooking Instructions"
        result = generic_graph.invoke({
            "llm": "llm",
            "task_name": task_name,
            "question": question,
            "simple_question": question,
            "context": " ",
            "final_answer": " ",
            "image_filename": " ",
            "CSV_file": " ",
            "HTML_to_display": "None",
            "recipes": recipes,
        })
        resp = result["final_answer"]
        print("RESPONSE: ", resp,"\n")
        recipes.append(resp)
        cumulative_menu_pages = cumulative_menu_pages + resp
###### END OF MAKING RECIPE PAGEFOR IDEA

############## MAKE A WINE PAIRING FOR RECIPE
        task_name = f"Summarize flavors of Recipe {i+1} and pair with wine"
        question = "Identify the flavors in this recipe and then pair with a wine from the wine list. Simply provide the words 'Selected Wine: ' followed by the selected wine, and why the selection was made - all formatted as HTML large font as per this example: Here is an example HTML format: <h1>Suggested Wine Pairing</h1><ul><li>Name of wine...</li><li>Explanation of flavor pairing rationale...</li></ul>. Here is the the recipe from a web page, followed by the wine list in CSV format."
        filename = ideas_so_far[i+1] + "html"
        with open(filename, "r") as file:
            context = file.read()

#        context = ideas_so_far[i+1] # was recipes[i+1]
        file = "Wine_list.txt"
        with open(output_path+file, 'r') as this_file:
             context += this_file.read()
        result = generic_graph.invoke({
            "llm": "normal",
            "task_name": task_name,
            "question": question,
            "simple_question": question,
            "context": context,
            "final_answer": " ",
            "HTML_to_display": "None",
        })
        resp = result["final_answer"]
        print("RESPONSE: ", resp,"\n")
        cumulative_menu_pages = cumulative_menu_pages + resp
###### END OF WINE PAIRING FOR RECIPE

    with open("IDEAS_AND_RECIPES.html", "w") as file:
        file.write(cumulative_menu_pages)

    return(True)

####++++++++++++++++++++++++ CSV question 
def csv_question_0() :
    import pandas as pd

    global TX_char, TX_tokn, RX_char, RX_tokn
    global left, right

#### WORKS GREAT - BUT HAVE TO TRIM INPUT LENGTH OR LOCK UP LAPTOP
    question = "I have a spreadsheet of restaurant order data and I want to know 'How many of each Main Course item were sold?' Format your answer as HTML page with large text."

#### 
    question = "I have a spreadsheet of restaurant order data and I want to know 'How many items were sold, where the Type is Main Course, broken out into individual Items?' Format your answer as HTML page with extra large text."

    image_filename = "data_visualization.png"
    CSV_file = "orders.csv"

    task_name = "Give AI a huge spreadsheet and see if it can count specific items within it."

    file = CSV_file
    ## Load dataset
    filepath = "./data/" + file
    with open(filepath, 'r', encoding='utf-8') as this_file:
        context = this_file.read()

    result = generic_graph.invoke({
        "llm": "llm",
        "task_name": task_name,
        "question": question,
        "simple_question": question,
        "context": context,
        "final_answer": " ",
        "image_filename": image_filename,
        "CSV_file": "orders.csv",
        "HTML_to_display": "orders.html",
    })
    resp = result["final_answer"]
    print("RESPONSE: ", resp,"\n")
    return(True)

####++++++++++++++++++++++++ CSV question to code
def csv_question_1() :
    import pandas as pd

    global TX_char, TX_tokn, RX_char, RX_tokn
    global left, right

    question = "I have a spreadsheet of restaurant order data and I want to know 'How many items were sold, where the Type is Main Course, broken out into individual Items?' Format your data visualization as a pie chart."

#### WORKS GREAT!
    question = "First, filter out only those items whose 'Type' is Main Course. Next, count how many of each 'Item' are in the list. Finally, create a pie chart that displays this information. Annotate the pie chart with the Item names."


    image_filename = "data_visualization.png"
    CSV_file = "orders.csv"

    task_name = "Write software to analyze a spreadsheet and create a chart."

    prompt_runnable = "You are an expert programmer working in a Python environment with 'from matplotlib import pyplot as plt' and 'import pandas as pd' already executed, and a CSV file has already been imported as a Pandas DataFrame named 'df' with the reply to Python command 'df.head()' being {head_of_CSV}. I want you to write a simple Python program that will be executed using the 'exec()' function to create the visualization requested in the HUMAN QUESTION: {question} \n and save it as a .png file at the location: ./data/{image_filename}. There is no need to use plt.show() since we will used the saved image file. Also, do not use '%matplotlib inline' as this is not needed. Only reply with the Python code as a string that will be sent to the 'exec()' function. Do not reply with any introductory or explanatory text - only reply with the Python code."

    prompt_runnable = "You are an expert programmer working in a Python environment with 'from matplotlib import pyplot as plt' and 'import pandas as pd' already executed, and a CSV file has already been imported as a Pandas DataFrame named 'df' with the reply to Python command 'df.head()' being {head_of_CSV}. I want you to write a simple Python program that will be executed using the 'exec()' function to create the visualization requested in the HUMAN QUESTION: {question} \n and save it as a .png file at the location: ./data/{image_filename}. There is no need to use plt.show() since we will used the saved image file. Also, do not use '%matplotlib inline' as this is not needed. Only reply with the Python code as a string that will be sent to the 'exec()' function. Do not reply with any introductory or explanatory text - only reply with the Python code."


    file = CSV_file
    ## Load dataset
    filepath = "./data/" + file
    df = pd.read_csv(filepath)
    head_of_CSV = str(df.head())

    csv_question_to_code_prompt = prompt_runnable.format(
        head_of_CSV = head_of_CSV, 
        image_filename = image_filename,
        question = question,
    )

    result = generate_plot_graph.invoke({
        "llm": "code",
        "task_name": task_name,
        "question": csv_question_to_code_prompt,
        "simple_question": question,
        "context": " ",
        "final_answer": " ",
        "image_filename": image_filename,
        "CSV_file": "orders.csv",
        "HTML_to_display": "orders.html",
    })
    resp = result["final_answer"]
    print("RESPONSE: ", resp,"\n")
    return(True)


####++++++++++++++++++++++++ CSV question to code
def csv_question_2() :
    import pandas as pd

    global TX_char, TX_tokn, RX_char, RX_tokn
    global left, right

#### WORKS GREAT!
    question = "First, filter out only the items 'Grilled Salmon'. Next, count how many there are for each day. Finally, create line graph that plots counts per day."

    image_filename = "data_visualization.png"
    CSV_file = "orders.csv"

    task_name = "Write software to analyze a spreadsheet and create a chart."

    prompt_runnable = "You are an expert programmer working in a Python environment with 'from matplotlib import pyplot as plt' and 'import pandas as pd' already executed, and a CSV file has already been imported as a Pandas DataFrame named 'df' with the reply to Python command 'df.head()' being {head_of_CSV}. I want you to write a simple Python program that will be executed using the 'exec()' function to create the visualization requested in the HUMAN QUESTION: {question} \n and save it as a .png file at the location: ./data/{image_filename}. There is no need to use plt.show() since we will used the saved image file. Also, do not use '%matplotlib inline' as this is not needed. Only reply with the Python code as a string that will be sent to the 'exec()' function. Do not reply with any introductory or explanatory text - only reply with the Python code."


    file = CSV_file
    ## Load dataset
    filepath = "./data/" + file
    df = pd.read_csv(filepath)
    head_of_CSV = str(df.head())

    csv_question_to_code_prompt = prompt_runnable.format(
        head_of_CSV = head_of_CSV, 
        image_filename = image_filename,
        question = question,
    )

    result = generate_plot_graph.invoke({
        "llm": "code",
        "task_name": task_name,
        "question": csv_question_to_code_prompt,
        "simple_question": question,
        "context": " ",
        "final_answer": " ",
        "image_filename": image_filename,
        "CSV_file": "orders.csv",
        "HTML_to_display": "orders.html",
    })
    resp = result["final_answer"]
    print("RESPONSE: ", resp,"\n")
    return(True)

####++++++++++++++++++++++++ ENGLISH TO SPANSIH
def translate_to_Spanish_good() :
    task_name = "Translate Recipe into Spanish"
    question = "Translate this recipe from English into Spanish. Here is the recipe from a web page."
    file = "Salmon recipe.txt"
    with open(output_path+file, 'r') as this_file:
        context = this_file.read()
    result = generic_graph.invoke({
        "llm": "normal",
        "task_name": task_name,
        "question": question,
        "simple_question": question,
        "context": context,
        "final_answer": " ",
        "HTML_to_display": "None",
    })
    resp = result["final_answer"]
    print("RESPONSE: ", resp,"\n")
    return(True)

####++++++++++++++++++++++++ RECIPE TO CAPTIONED IMAGE
def recipe_to_captioned_image_invoke(main_course = "Salmon recipe") :
    global salmon_page
 
    recipe = main_course + '.txt'
    image_filename = main_course + '.png'
    task_name = "Recipe to Captioned Image"
    file = recipe
    with open(output_path+file, 'r') as this_file:
        context = this_file.read()

    recipe_to_captioned_image_prompt = recipe_to_captioned_image.format(
        context=context, 
        image_filename = image_filename)

    result = generate_image_graph.invoke({
        "llm": "normal",
        "task_name": task_name,
        "question": recipe_to_captioned_image_prompt,
        "simple_question": recipe_to_captioned_image_prompt,
        "context": context,
        "image_filename": image_filename,
        "final_answer": " ",
        "HTML_to_display": "None",
    })
    resp = result["final_answer"]
    with open(salmon_page, "w") as file:
        file.write(resp)
    print("RESPONSE: ", resp,"\n")
    return(True)

####++++++++++++++++++++++++ Wine Pairing
def Wine_Paring() :
    task_name = "Summarize flavors of recipe and pair with wine"
    question = "Identify the flavors in this recipe and then pair with a wine from the wine list. Simply provide the recipe name, the selected wine, and why the selection was made - all in an easy to read HTML page using extra large font. Here is the the wine list in CSV format, followed by the recipe from a web page."
    file = "Wine_list.txt"
    with open(output_path+file, 'r') as this_file:
        context = this_file.read()
    file = "Salmon recipe.txt"
    with open(output_path+file, 'r') as this_file:
        context += this_file.read()
    result = generic_graph.invoke({
        "llm": "normal",
        "task_name": task_name,
        "question": question,
        "simple_question": question,
        "context": context,
        "final_answer": " ",
        "HTML_to_display": "None",
    })
    resp = result["final_answer"]
    print("RESPONSE: ", resp,"\n")
    return(True)


####++++++++++++++++++++++++ ENGLISH TO JSON
def translate_to_JSON() :
    global left, right

    task_name = "Translate Recipe into JSON Format for a Database"
    question = "I want to put this website recipe into my database. Can you structure it into JSON with objects 'name', 'ingredients', and 'procedure' for me?"
    file = "Salmon recipe.txt"
    with open(output_path+file, 'r') as SOAP:
        context = SOAP.read()
    result = generic_graph.invoke({
        "llm": "normal",
        "task_name": task_name,
        "question": question,
        "simple_question": question,
        "context": context,
        "final_answer": " ",
        "HTML_to_display": "None",
    })
    resp = result["final_answer"]
    print("RESPONSE: ", resp,"\n")
    return(True)


####++++++++++++++++++++++++ CSV TO HTML
def CSV_to_html() :
    global left, right

    task_name = "Translate from Spreadsheet to Readable format"
    question = "Convert this CSV into HTML format and use extra large font for readability."
    file = "Culinary Spreadsheet 2 (Inventory).csv"
    with open(output_path+file, 'r') as SOAP:
        context = SOAP.read()
    result = generic_graph.invoke({
        "llm": "normal",
        "task_name": task_name,
        "question": question,
        "simple_question": question,
        "context": context,
        "final_answer": " ",
        "HTML_to_display": "None",
    })
    resp = result["final_answer"]
    print("RESPONSE: ", resp,"\n")
    return(True)


####++++++++++++++++++++++++ CSV QUERY 
def CSV_query() :
    global left, right

    task_name = "Ask questions about a Small Spreadsheet"
    question = """
        In this inventory, what proteins/meats are the most expensive per pound? 
        Format your response as an easy to read HTML table.
        Use an extra-large font for readability.
    """
    file = "Culinary Spreadsheet 2 (Inventory).csv"
    with open(output_path+file, 'r') as SOAP:
        context = SOAP.read()
    result = generic_graph.invoke({
        "llm": "normal",
        "task_name": task_name,
        "question": question,
        "simple_question": question,
        "context": context,
        "final_answer": " ",
        "HTML_to_display": "None",
    })
    resp = result["final_answer"]
    print("RESPONSE: ", resp,"\n")
    return(True)


####++++++++++++++++++++++++ IMAGE TO CSV
def image_to_csv() :
    global left, right, page_w, TX_char, TX_tokn, RX_char, RX_tokn

    task_name = "Now using a VLM for Multi-Modal Analysis - Simple 'llava' model that is 4.7 GB"
    task_details1 = "Convert the purchase order items in this image into tabular HTML format with headings 'Ingredient', 'Additional Detail', 'Units', 'Quantity', and 'Price'. Try to fill the fields with the info from the image. Only reply with the an HTML page of the nicely formatted purchase order information and no other text."
#    task_details2 = '<img src="./data/INVOICE.png">'
    task_details2 = '<img src="./data/INVOICE-zoom.png" alt="Invoice zoom-in"' + 'width=' + str(page_w-10) + '>'
    # Display the next step in the storyboard
    html_to_png("<body><h1>" + task_name + "</h1><h2>" + "Prompt: " + task_details1  + "</h2>" +  task_details2, output_path=left)
    update_display(left = left, right = False)
    
    ### VISUAL LANGUAGE MODEL SETUP
    # This is the LLM model (really a VLM) that is able to take images as well as text for input
    from langchain_ollama import OllamaLLM
#    visual_llm = OllamaLLM(model="gemma3:12b-it-qat", temperature = 0)
    visual_llm = OllamaLLM(model="llava", temperature = 0)

    # Photo file handling portion
    import base64
    from io import BytesIO
    from IPython.display import HTML, display
    from PIL import Image
    # Load up a sample photo
#    photo_file = ".\data\INVOICE.png"
    photo_file = ".\data\INVOICE-zoom.png"
    pil_image = Image.open(photo_file)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # You can change the format if needed
    image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    visual_llm_with_image_context = visual_llm.bind(images=[image_b64])

    char, tokn = token_len(task_details1)
    if tokn > num_ctx : print(f"\n-->WARNING Token count {tokn} exceeded max context token limit of {num_ctx}")
    TX_char += char
    TX_tokn += tokn + 256 # 256 is a typical input image token size

    response = visual_llm_with_image_context.invoke(task_details1)

    char, tokn = token_len(task_details1)
    RX_char += char
    RX_tokn += tokn
    
    print("RESPONSE: ", response,"\n")
    
    # Display the next step in the storyboard
    left = output_path + left_page
    right = output_path + right_page
    html_to_png(response, output_path=right)
    update_display(left, right)

    return(True)

####++++++++++++++++++++++++ BETTER PROMPTED IMAGE TO CSV
def better_image_to_csv() :
    global left, right, page_w, TX_char, TX_tokn, RX_char, RX_tokn

    task_name = "Now using a more advanced VLM - 'gemma3:12b-it-qat' that is 8.9 GB."
    task_details1 = "Convert the purchase order items in this image into tabular HTML format with headings 'Ingredient', 'Additional Detail', 'Units', 'Quantity', and 'Price'. Try to fill the fields with the info from the image. Only reply with the an HTML page of the nicely formatted purchase order information and no other text."

#    task_details2 = '<img src="./data/INVOICE.png">'
    task_details2 = '<img src="./data/INVOICE-zoom.png" alt="Invoice zoom-in"' + 'width=' + str(page_w-10) + '>'
    # Display the next step in the storyboard
    html_to_png("<body><h1>" + task_name + "</h1><h2>" + "Prompt: " + task_details1  + "</h2>" +  task_details2, output_path=left)
    update_display(left = left, right = False)
    
    ### VISUAL LANGUAGE MODEL SETUP
    # This is the LLM model (really a VLM) that is able to take images as well as text for input
    from langchain_ollama import OllamaLLM
    visual_llm = OllamaLLM(model="gemma3:12b-it-qat", temperature = 0)
#    visual_llm = OllamaLLM(model="llava", temperature = 0)

    # Photo file handling portion
    import base64
    from io import BytesIO
    from IPython.display import HTML, display
    from PIL import Image
    # Load up a sample photo
#    photo_file = ".\data\INVOICE.png"
    photo_file = ".\data\INVOICE-zoom.png"
    pil_image = Image.open(photo_file)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # You can change the format if needed
    image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    visual_llm_with_image_context = visual_llm.bind(images=[image_b64])

    char, tokn = token_len(task_details1)
    if tokn > num_ctx : print(f"\n-->WARNING Token count {tokn} exceeded max context token limit of {num_ctx}")
    TX_char += char
    TX_tokn += tokn + 256 # 256 is a typical input image token size

    response = visual_llm_with_image_context.invoke(task_details1)

    char, tokn = token_len(task_details1)
    RX_char += char
    RX_tokn += tokn
    
    print("RESPONSE: ", response,"\n")
    
    # Display the next step in the storyboard
    left = output_path + left_page
    right = output_path + right_page
    html_to_png(response, output_path=right)
    update_display(left, right)

    return(True)

####++++++++++++++++++++++++ Sentiment Analysis
def sentiment_analysis() :
    global left, right

    task_name = "Determine Sentiment and Extract Semantic Meaning"

    question = "Read through customer reviews and see if you can find information that might reveal why Grilled Salmon sales have gone gradually downward. Also, provide a summary of customer recommendations, if you find any. Format your response as an HTML page that is neatly organized with extra large font. Here are the reviews:"

    file = "customer_reviews.html"
    with open(output_path+file, 'r', encoding='utf-8') as SOAP:
        context = SOAP.read()
    result = generic_graph.invoke({
        "llm": "normal",
        "task_name": task_name,
        "question": question,
        "simple_question": question,
        "context": context,
        "final_answer": " ",
        "HTML_to_display": "None",
    })
    resp = result["final_answer"]
    file = "business_recommendation.html"
    with open(output_path+file, 'w', encoding='utf-8') as file:
        file.write(resp)
    print("RESPONSE: ", resp,"\n")
    return(True)


####==========================================================================####
####/////////////////      END OF DEMO FUNCTIONS     /////////////////////////####
####==========================================================================####

###############################################################################
######################### RUN SECTION #########################################
###############################################################################
print("""
#############################
#### Run the slideshow from start to finish
#############################
""")
########## SETTING UP DISPLAY CONSTANTS, ARROW CONTROL, AND PRESENTATION SEQUENCE
#### Utility functions in the same directory
# NEEDED? from FFC_utils import SlideShow, update_display, setup_screen

# setup SLIDESHOW constants and setup screen

def no_op() : return(True)


###########################################################
# Slide show arrow handling and run loop
###########################################################

def await_arrow(presentation: SlideShow, demo: bool = False, demo_delay: int = 3) -> bool:
    """
    Wait for left or right arrow key press.

    If demo is True, auto-press right after demo_delay seconds.

    Returns:
        True if we moved to another slide, False if at the start or end.
    """
    global TX_char, TX_tokn, RX_char, RX_tokn

    pygame.event.clear()
    start_time = time.time()

    while True:
        if demo and (time.time() - start_time) >= demo_delay:
            print("[DEMO] --> RIGHT ARROW AUTO-PRESSED")
            TX_char = TX_tokn = RX_char = RX_tokn = 0
            return presentation.forward()

        events = pygame.event.get()
        for e in events:
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RIGHT:
                    print("--> RIGHT ARROW PRESSED")
                    TX_char = TX_tokn = RX_char = RX_tokn = 0
                    return presentation.forward()
                if e.key == pygame.K_LEFT:
                    print("<-- LEFT ARROW PRESSED")
                    TX_char = TX_tokn = RX_char = RX_tokn = 0
                    return presentation.backward()


def run_demo() -> None:
    """
    Top-level function to run the Chef's Buddy slideshow demo.

    - Initializes the screen and slideshow
    - Wires in the demo functions for selected slide indices
    - Enters the left/right arrow event loop
    """
#    from FFC_utils import presentation  # populated by setup_screen

    global left, right, page_w, TX_char, TX_tokn, RX_char, RX_tokn

    import FFC_utils

    print(
        """
#############################
#### Run the slideshow from start to finish
#############################
"""
    )

    # Wire demo functions for slides that use LLMs
    from FFC_utils import no_op  # if you want some placeholder slides

    llm_functions = [
        translate_to_Spanish_good,
        Wine_Paring,
        translate_to_JSON,
        recipe_to_captioned_image_invoke,
        csv_question_0,
        csv_question_1,
        csv_question_2,
        sentiment_analysis,
        image_to_csv,
        better_image_to_csv,
        recipe_ideas_function,
        no_op,
        no_op,
        no_op,
        no_op,
        no_op,
        no_op,
        no_op,
        no_op,
        no_op,
        no_op,
        no_op,
    ]

    FFC_utils.setup_screen(llm_functions)

    from FFC_utils import (
        monitor_width,
        monitor_height,
        X,
        Y,
        font_size,
        page_w,
        page_h,
    )

    print(
        "monitor_width, monitor_height, X, Y, font_size, page_w, page_h,",
        monitor_width,
        monitor_height,
        X,
        Y,
        font_size,
        page_w,
        page_h,
    )

    update_display(left=None, right=None)

    more_slides = True
    while more_slides:
        more_slides = await_arrow(FFC_utils.presentation, demo=demo, demo_delay=demo_delay)

    close_down_show()


if __name__ == "__main__":
    run_demo()


# ---===---===---===---===---===---===---===---===
#     INSTRUCTIONS ON HOW TO INSTALL FOR WINDOWS WITH GPU:
# Python 3.12  instead of # Install the Python version (3.10.6) from here: https://www.python.org/downloads/windows/
# Install pip as follows: py -m pip install pip --upgrade
# Install CUDA as follows: py -m pip install --verbose nvidia-cuda-runtime-cu11
# pip install torch
# Install PyTorch (torch) that uses CUDA as follows: python -m pip install --verbose torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
# Make sure the python scripts are in the path
# Press the Start key and search for “Edit the system environment variables”.
# Go to the “Advanced” tab and click the “Environment variables” button.
# Select the “Path” variable under “User variables” or “System variables”.
# Click the “Edit” button and press “New” to add the desired path.
# C:\Users\EET\AppData\Local\Programs\Python\Python310\Scripts
# ---===---===---===---===---===---===---===---===
# Then confirm that Torch was indeed installed and it is set up to use CUDA, buy doing the following within python:
# python
# >>> import torch
# 
# >>> print("Torch version:",torch.__version__)
# Torch version: 2.1.0+cu118
# 
# >>> print("Is CUDA enabled?",torch.cuda.is_available())
# Is CUDA enabled? True
# ---===---===---===---===---===---===---===---===# 
#     GOOGLE MODELS TO DOWNLOAD FROM HUGGINGFACE
#    THESE GO IN "Models" FOLDER
# v1-5-pruned-emaonly.safetensors
# v2-1_768-ema-pruned.safetensors
#    THIS GOES IN A NEW "Upscaler" FOLDER (or you can stick in the models folder if you're not using AUTOMATIC1111 GUI)
# 4x_UniversalUpscalerV2-Sharp_101000_G.pth
#
# Then change the file locations here at the top of this python script:
# models = "C:/Users/EET/Desktop/Culinary Slideshow/models"
# ---===---===---===---===---===---===---===---===
#     INSTRUCTIONS ON HOW TO INSTALL PYGAME TO SHOW THINGS ON THE SCREEN
# install pygame using pip "pip install pygame"
# ---===---===---===---===---===---===---===---===
#     INSTRUCTIONS ON HOW TO INSTALL OLLAMA and LLAMA3 8B FOR WINDOWS WITH GPU:
# Install for Windows from here: https://ollama.com/download/windows
# Run the installer
# Pull the desired model. We will use the 8B parameter llama3 using the command line
# ollama pull llama3.1:8b
# You can test it from the command line using "ollama run llama3.1:8b"
# install ollama using pip "pip install ollama"
# ---===---===---===---===---===---===---===---===
# Installing automatic1111
# https://github.com/AUTOMATIC1111/stable-diffusion-webui
# Installation on Windows 10/11 with NVidia-GPUs using release package
# Download sd.webui.zip from v1.0.0-pre (https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
# Run update.bat.
# Run run.bat.
# For more details see Install-and-Run-on-NVidia-GPUs here: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs
# ---===---===---===---===---===---===---===---===
# To run this script, first start up the ollama web server
# "ollama run llama3.1:8b" from the command line
# Will also need to run a VLM for image analysis and OCR
# "ollama run llava"
# You can try out a few chat responses, and then type "/bye" to exit the text interface but leaves the server running
# ---===---===---===---===---===---===---===---===
# After starting up the ollama server, we need to start up the stable diffusion server to create images.
# You may need to edit the "webui_user.bat" in the stable-diffusion-webui directory. 
# Edit that file to add the --api option Here is the uptades batch file
#@echo off
#set PYTHON=
#set GIT=
#set VENV_DIR=
#set COMMANDLINE_ARGS= --api
#call webui.bat
#
# and then run that batch file from the command line. Make a note of the URL and Port number
# In the case of this computer, it is "http://127.0.0.1:7860/"
# "webui_user.bat"
# You can play around and generate an image or two.
# Then close the browser window, as we'll use the API and keep the server running
# More info on using the API https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API
# ---===---===---===---===---===---===---===---===
# After running it the first time to download Hugging Face embeddings, this will run without internet connection
# Always first boot up ollama server --> ollama run llama3.1:8b --> /bye  to end interactice session.
# ---===---===---===---===---===---===---===---===
########## SETTING UP LANGCHAIN
# Trying langchain
# ### ollama run llama3.1:8b
# pip install langchain
# pip install langchain-ollama
######## The below line messes up the Numpy version conflicting with Torch
# pip install langchain-community
########
# pip install langchain-experimental
# pip install faiss-cpu
# pip install pypdf2
# lc18
# pip install langgraph
### Added for using OpenAI wrapper around Ollama
# pip install langchain_openai
# Added IPython for graph diagram display
# pip install IPython
#
# To be able to convert HTML files into Images for display 
# pip install selenium webdriver-manager
# Also, make sure the Chrome browser has been installed (latest version as of May 2025)
#
# I have not had to set up these ENVIRONMENT VARIABLES
# set LANGSMITH_TRACING="false"
# In Windows, search for "environment" and edit the environment variables
# Add a new SYSTEM variable as follows:
# set KMP_DUPLICATE_LIB_OK=True
#
# lc05 swapping html2image (worked with old Chromium) with Selenium
# pip install selenium webdriver-manager
# v02 - use ./data as directory
#
# NOT SURE IF THE BELOW ARE NEEDED
#   for Microsoft word docx files
# pip install docx2txt         
#   Also convert all PPTX files (PowerPoint) to PDF or it will crash on .pptx
#   Finally, for Excel documents use
# pip install openpyxl
########################
