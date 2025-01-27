# 🤖 Chat with multiple PDFs locally

![alt text](assets/demo.png)

# 📖 Table of Contents

- [`Feature`](#⭐️-features)
- [`Idea`](#-idea)
- [`Setup`](#💻-setup)
  - [`Kaggle`](#1-kaggle-recommended)
  - [`Local`](#2-local)
    - [`Clone`](#21-clone-project)
    - [`Install`](#22-install)
    - [`Run`](#23-run)
- [`Todo`](#🎯-todo)

# ⭐️ Key Features

- Easy to run on `Local` or `Kaggle` (new)
- Using any model from `Huggingface` and `Ollama`
- Process multiple PDF inputs.
- Chat with multiples languages (Coming soon).
- Simple UI with `Gradio`.

# 💡 Idea (Experiment)

![](./assets/rag-flow.svg)

![](./assets/retriever.svg)

# 💻 Setup

## 1. Kaggle (Recommended)

- Import [`notebooks/kaggle.ipynb`](notebooks/kaggle.ipynb) to Kaggle
- Replace `<YOUR_NGROK_TOKEN>` with your tokens.

## 2. Local

### 2.1. Clone project

```bash
git clone https://github.com/datvodinh/rag-chatbot.git
cd rag-chatbot
```

### 2.2 Install

#### 2.2.1 Docker

```bash
docker compose up --build
```

#### 2.2.2 Using script (Ollama, Ngrok, python package)

```bash
source ./scripts/install_extra.sh
```

#### 2.2.3 Install manually

##### 1. `Ollama`

- MacOS, Window: [Download](https://ollama.com/)

- Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

##### 2. `Ngrok`

- Macos

```bash
brew install ngrok/ngrok/ngrok
```

- Linux

```bash
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
| sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
&& echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
| sudo tee /etc/apt/sources.list.d/ngrok.list \
&& sudo apt update \
&& sudo apt install ngrok
```

##### 3. Install `rag_chatbot` Package

```bash
source ./scripts/install.sh
```

### 2.3 Run

```bash
source ./scripts/run.sh
```

or

```bash
python -m rag_chatbot --host localhost
```

- Using Ngrok

```bash
source ./scripts/run.sh --ngrok
```

### 3. Go to: `http://0.0.0.0:7860/` or Ngrok link after setup completed

## 🎯 Todo

- [x] Add evaluation.
- [x] Better Document Processing.
- [ ] Support better Embedding Model for Vietnamese and other languages.
- [ ] ReAct Agent.
- [ ] Document mangement (Qrdant, MongoDB,...)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=datvodinh/rag-chatbot&type=Date)](https://star-history.com/#datvodinh/rag-chatbot&Date)


### Additions for DSP from Yari:
In the current form, not all packages are explicitly compatible with each other, thus we use a workaround to get all the packages working.

WARNING: this can break your Python environment. Please use a virtual environment.

#### To install everything properly:
0. Install Python 3.11, version 3.12 is currently not supported by Llama-index and may lead to errors
1. Ensure that you have installed poetry
2. Set the working directory in your terminal to this folder
3. Run poetry install to install the dependencies
4. Run `pip install llama-index-graph-stores-neo4j`
5. Run `pip install llama-index -U`
5. Run `pip install llama-index-core -U`

There will be errors shown that  pip is not taking into account all packages, as not all packages are explicitly compatible with each other. This message may be ignored, as the chatbot will run without issue. If you get any errors from huggingface, make sure to delete the cache folder first before trying anything else.