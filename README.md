# MindCoders Chatbot

**MindCoders Chatbot** is a lightweight, Python-based chatbot that uses [describe the tech used — e.g. a language model, embeddings, etc., update as needed] to generate conversational responses.


## Features

- Quick and easy conversational interface via the terminal
- Customizable response behavior (update with specifics)
- Modular loader component for data or model inputs

---

## Getting Started

### Prerequisites

- Python 3.11+  
- (Optionally) A virtual environment like `venv` or `conda`

### Installation

1. Clone the repo:

    ```bash
    git clone https://github.com/UdayMalviya/mindcoders_chatbot.git
    cd mindcoders_chatbot
    ```

2. Set up a virtual environment and install dependencies:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # or `venv\Scripts\activate` on Windows
    pip install -r requirement.txt
    ```
3. Set env variables :
   1. **GOOGLE_API_KEY** = Your api key from [api](https://ai.google.dev/gemini-api/docs)
   2. **LANGSMITH_API_KEY** = Your api key from the [website](https://smith.langchain.com/)
   3. **USER_AGENT**
   4. **LANGSMITH_TRACING** = The value must be set to **TRUE**

---

## Usage

Run the chatbot:

```bash
python chat.py
