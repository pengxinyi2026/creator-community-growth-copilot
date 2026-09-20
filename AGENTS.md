# AI-Powered Creator Community Growth Copilot

## Project purpose

This project is an independent portfolio project. It demonstrates creator community operations, creator growth, campaign matching, social commerce, data analysis, AI-assisted automation, and reporting.

## Project owner

The project owner is a complete Python beginner. Prefer explanations, code, and file structures that are easy to read and learn from.

## Technical stack

- Python 3
- pandas
- numpy
- Streamlit
- OpenAI Python SDK
- python-dotenv
- Altair

## Core architecture

Synthetic creator data → validation → metrics → creator scoring → segmentation → campaign matching → AI insights → reporting → Streamlit dashboard

## Engineering principles

- Use simple, readable Python.
- Keep functions small and use clear variable names.
- Add comments for important logic.
- Make scoring rules transparent and explainable.
- Keep creator-facing communication human-in-the-loop: people review and approve AI drafts before use.
- Do not introduce black-box machine learning unless explicitly requested.
- Do not add unnecessary frameworks.

## Security and data rules

- Read API keys from environment variables, never from source code.
- Never hard-code credentials or commit a `.env` file.
- Never use real private creator data.
- Use synthetic creator data only.
- Do not scrape TikTok or use real TikTok internal data.

## Portfolio disclaimer

This is an independent portfolio project using synthetic data and is not affiliated with TikTok.
