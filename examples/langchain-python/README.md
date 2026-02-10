# LangChain + Blindfold (Python)

PII-safe LangChain chains using `RunnableLambda`. Wraps any chain with Blindfold tokenization/detokenization so PII never reaches the language model.

## How it works

Uses LangChain's composable pipeline:

```
tokenize → prompt → LLM → detokenize
```

The `blindfold_protect()` helper returns two `RunnableLambda` steps that plug into any chain.

## Setup

```bash
pip install -r requirements.txt
cp ../../.env.example .env
# Edit .env with your API keys
```

## Run

```bash
python main.py
```
