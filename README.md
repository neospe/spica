
<img src="assets/spica-icon-512.png" width="200">

# spica

minimalistic multi-platform UI for local LLMs.

- interact with any language model in GGUF format
- edit mode for chats: quickly condense parts of the interaction
- easily summarize and manage sessions
- manage generation parameters and constraints
- light and dark themes
- minimal footprint and dependencies (all in 50mb)

# download

there are pre-built binaries for **macos**, **windows**, and **linux**. head over to the releases tab.

# quick start

## choose a model file

you can use any model available in GGUF format. most current models can be found here: https://huggingface.co/TheBloke

the LLM leaderboard helps you choose the best model for your requirements: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

download and set the model binary as `model file` under settings.

## set model parameters

a few parameters need to be set according to the information in the model card.

for example, [dolphin/mistral-7B](https://huggingface.co/TheBloke/dolphin-2.1-mistral-7B-GGUF#prompt-template-chatml) works well using the following values:

- context size: 4096
- system prompt: `<|im_start|>system\nThis is a system prompt, please behave and help the user.<|im_end|>`
- start user tag: `<|im_start|>user `
- start assistant tag: `<|im_start|>assistant `

ending tags are also possible, but not always needed (as in this example).

# keyboard shortcuts

- enter message: <kbd>ctrl</kbd>+<kbd>enter</kbd>
- change text size: <kbd>ctrl</kbd>+<kbd>+</kbd>/<kbd>-</kbd>
- chat edit mode
    - enter: click on text
    - de/select all: <kbd>ctrl</kbd>+<kbd>a</kbd>
    - exit: <kbd>esc</kbd>

**note**: on macos, use <kbd>cmd</kbd> respectively.

# optional settings

## tune generation parameters

improve generations for a particular text type quickly by tuning the `temperature` (range: 0.1-2.0) and `repetition penalty` (range: 1.0-2.0) values.

## add a constraining grammar

generate specific text structures like data or code using a constraining grammar:

- browse the collection: https://github.com/antlr/antlr4
- build your own: https://github.com/IntrinsicLabsAI/grammar-builder

# build

```
git clone https://github.com/hexis-systems/spica && cd spica
pip install --no-deps -r requirements.txt
flet pack spica.py --icon assets/spica-icon-512.png --name spica --add-binary=$PYTHONPATH/llama_cpp/libllama.dylib:llama_cpp
```

**note**: on macos you'll need a "framework" install of python. installers from python.org and homebrew are fine. if you're using `pyenv`, do the following:

```
PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install 3.10.9
```

# roadmap

- retrieval augmented generation
    - using _sqlite-vss_, asymmetric embeddings like _e5_, hybrid vector/kw-search
    - ingest common document formats, scrape web pages
- settings
    - gpu offloading parameters
    - prompt template management
    - define actions/chains of actions, similar to condensation
- sessions
    - fulltext search, sort, export
- ui
    - beautify LLM output and eventually add widgets
    - more themes

# license

MIT License