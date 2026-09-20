# AI-Powered Creator Community Growth Copilot

## Overview

This project is a beginner-friendly Python portfolio project for exploring how a creator community team could organise synthetic creator data, spot growth opportunities, and prepare useful reports. It will combine simple data analysis, optional AI assistance, and a Streamlit dashboard.

## Problem

Creator teams may need to review many creators at once. They need a practical way to analyse creator performance, identify growth opportunities, match creators to campaigns, find inactive creators who may be ready to return, prepare communication drafts, and produce recurring reports.

## Solution

The project uses Python and simple data analysis to turn a creator CSV file into clear metrics, scores, segments, recommendations, and reports. AI will be used only where it adds value, such as explaining data-driven findings or drafting outreach for human review. A repeatable pipeline will reduce repetitive manual work.

## Workflow

```text
Creator CSV
    ↓
Data validation
    ↓
Performance metrics
    ↓
Creator Growth Score
    ↓
Creator segmentation
    ↓
Campaign matching
    ↓
AI insights and outreach drafts
    ↓
Weekly reporting
    ↓
Streamlit dashboard
```

## Features

- Synthetic creator data generation
- Data validation and cleaning checks
- Creator performance metrics
- Transparent Creator Growth Score
- Creator segmentation
- Campaign-to-creator matching
- Identification of creators for possible reactivation
- AI-generated creator insights
- Personalised outreach drafts for human review
- Weekly community performance reports
- Streamlit dashboard
- Reproducible automated pipeline

## Tech Stack

- Python 3
- pandas and numpy for data analysis
- Streamlit for the dashboard
- Altair for charts
- OpenAI Python SDK for optional AI-assisted insights and drafts
- python-dotenv for reading local environment variables safely
- pyarrow for efficient data support

## Project Structure

```text
.
├── data/
│   ├── raw/                 # Synthetic input CSV files
│   └── processed/           # Cleaned and calculated output data
├── src/
│   ├── generate_data.py     # Future synthetic data generator
│   ├── validate_data.py     # Future data checks
│   ├── metrics.py           # Future performance metrics
│   ├── scoring.py           # Future Growth Score logic
│   ├── segmentation.py      # Future creator groups
│   ├── matching.py          # Future campaign matching
│   ├── ai_insights.py       # Future optional AI features
│   └── reporting.py         # Future weekly reports
├── app.py                   # Streamlit application entry point
├── run_pipeline.py          # Pipeline entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Safe environment-variable template
├── .gitignore               # Files Git must not track
└── AGENTS.md                # Instructions for future coding agents
```

## Data

- All creator data in this project is synthetic.
- No real TikTok internal data is used.
- No scraping is performed.

## Human-in-the-loop

AI may generate recommendations and outreach drafts, but it does not send messages or make final decisions. A person must review, edit, and approve every creator-facing communication before it is used.

## Disclaimer

This is an independent portfolio project using synthetic data. It is not affiliated with or connected to TikTok.
