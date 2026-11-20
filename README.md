# Shift-Change Request Agent

## Requirements

- Docker Desktop (with Docker Compose)
- A free Groq API key from https://console.groq.com/keys (only needed for the live chat and the real-model eval)

## 1. Configure

```bash
cp .env.example .env
```

Open `.env` and set `LLM_API_KEY=` to your Groq key. Change `JWT_SECRET` to any long random string.

## 2. Run the app

```bash
docker compose up --build
```

Open http://localhost:8000. Pick an employee on the left to chat, and a manager on the right to approve or reject requests.

To reset the demo data:

```bash
docker compose exec app python -m app.seed --reset
docker compose restart app
```

## 3. Run the tests and the evaluation set

No API key needed (uses a scripted fake LLM):

```bash
docker compose run --rm tests
```

This runs pytest and then `eval/run_eval.py`, which writes `eval/eval.jsonl` and `eval/summary.json`.

## 4. Run the evaluation with the real model (optional)

```bash
docker compose run --rm -e LLM_API_KEY=<your key> -e LLM_MODEL=openai/gpt-oss-20b \
  -e LLM_REASONING_EFFORT=low -e LLM_MAX_TOKENS=300 -e LLM_TIMEOUT_S=4.0 -e EVAL_PACE_S=7 \
  -v "$(pwd)/eval:/srv/eval" tests python -m eval.run_eval --llm real
```

Results go to `eval/eval_real_llm.jsonl` and `eval/summary_real_llm.json`. `EVAL_PACE_S=7` keeps the run under the Groq free tier's tokens-per-minute limit.

Held-out phrasings against the running app (needs step 2 running and Python with `httpx` installed):

```bash
python eval/heldout.py
```

## Without Docker

Needs Python 3.13 and PostgreSQL 16 running locally (`postgresql://postgres:postgres@localhost:5432/shift`, or set `DATABASE_URL`).

```bash
pip install -r requirements.txt
python -m app.seed
uvicorn app.main:app
pytest
```
