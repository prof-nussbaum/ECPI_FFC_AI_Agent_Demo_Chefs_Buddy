"""
FFC_Functions.py

Small utility module showing how to let an LLM write Pandas code against a local CSV.

This is intentionally self-contained:
- It sets up its own ChatOllama client and embeddings.
- It uses a local tokenizer.json to count tokens.
- It works with a CSV named 'Inventory.csv' in the current directory.
""" 

from __future__ import annotations

from typing import Tuple

from langchain_ollama import ChatOllama, OllamaEmbeddings
from transformers import PreTrainedTokenizerFast

#############################
# LLM / embeddings setup
#############################

# Pretty fast, good answers
LOCAL_LLM_MODEL = "llama3.2:3b"
NUM_CTX = 4096  # context token limit for this small demo

print(
    """
#############################
#### SETUP INFERENCE AND ENVIRONMENT HYPERPARAMETERS
#############################
"""
)
TX_char = 0
TX_tokn = 0
RX_char = 0
RX_tokn = 0
print(
    f"# running count of token TX/RX {TX_tokn}/{RX_tokn} and characters "
    f"TX to LLM, RX received back from LLM {TX_char}/{RX_char}"
)
print(f"# Context token count maximum (contexts+questions can get truncated if longer {NUM_CTX})")
print(f"# Model Selection - Ollama serving {LOCAL_LLM_MODEL}")

print(
    """
#############################
#### SETUP LLM(s)
#############################
"""
)

# num_ctx is context max tokens
llm = ChatOllama(model=LOCAL_LLM_MODEL, temperature=0, num_ctx=NUM_CTX)
llm_json_mode = ChatOllama(
    model=LOCAL_LLM_MODEL,
    temperature=0,
    format="json",
    num_ctx=NUM_CTX,
)

embeddings = OllamaEmbeddings(model="llama3")

#############################
#### Tokenizer / token counting
#############################

llama_tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer.json",
    clean_up_tokenization_spaces=True,
)


def token_len(text: str) -> Tuple[int, int]:
    """
    Count characters and tokens of the given text using the local tokenizer.json.
    """
    tokens = len(llama_tokenizer.encode(text=text))
    characters = len(text)
    return characters, tokens


#############################
#### Pandas + LLM demo
#############################

def pandas_codewriter() -> bool:
    """
    Let the LLM write a Pandas snippet over Inventory.csv.

    This function:
    - Loads ./Inventory.csv into a DataFrame df
    - Constructs a natural-language question asking for code to sum the
      'Total Value' column
    - Asks the LLM to return *only* a Python code snippet
    - Prints that code snippet

    Returns:
        True always, to signal to the slideshow that it handled its own display.
    """
    from langchain_core.tools import Tool
    from langchain_experimental.utilities import PythonREPL
    import pandas as pd

    python_repl = PythonREPL()
    Tool(
        name="python_repl",
        description=(
            "A Python shell. Use this to execute python commands. "
            "Input should be a valid python command. If you want to see the "
            "output of a value, you should print it out with `print(...)`."
        ),
        func=python_repl.run,
    )

    df = pd.read_csv("Inventory.csv")

    # Final, simplest prompt (matches original)
    q = (
        "I have a pandas DataFrame 'df' with column 'Total Value'. "
        "Write code to compute the sum of all the dollar amounts in the "
        "'Total Value' column. Do not assume access to any libraries, queries, "
        "database functions, or any other methods besides standard Python and "
        "the Pandas library as 'pd'. Return a Python code snippet and nothing else."
    )

    ai_msg = llm.invoke(q)
    print(ai_msg.content)
    return True


if __name__ == "__main__":
    # Simple local test: just run the codewriter.
    pandas_codewriter()
