# Reflexion Agent

![alt text](image.png)

Reflexion Agent is a pattern where the Actor agent will generate a draft with an answer, search term. Rivisor Agent will review such draft several times until a good quality is reached.

## How it works


## Environment variables

Create a `.env` file and fill in:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | From [platform.openai.com](https://platform.openai.com) |
| `LANGSMITH_API_KEY` | From [langchain.com/langsmith](https://www.langchain.com/langsmith) |
| `LANGSMITH_PROJECT` | Your LangSmith project name |
| `TAVILY_API_KEY` | From [app.tavily.com](https://app.tavily.com) |

## Usage

### 1. Install dependencies

```bash
uv sync
```

### 2. Run ingestion

```bash
uv run python main.py
```
