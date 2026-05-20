---
title: Pharma Intelligence
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
---

# 🧬 PatSnap Pharma Intelligence

**AI-Powered Drug Discovery Intelligence Agent** — explore targets, drugs, diseases, companies, and clinical trials.

## Modules

| Module | Description |
|--------|-------------|
| 🤖 **Agent Chat** | Natural language → intent routing → structured report |
| 🎯 **Target Intelligence** | Deep analysis of drug targets (biology, drugs, pipeline) |
| 💊 **Drug Exploration** | Pipeline drugs by target, disease, mechanism, or company |
| 🏥 **Disease Investigation** | Disease landscape, epidemiology, treatments |
| 🏢 **Company Profiling** | Pharma pipeline & strategic analysis |
| 🧪 **Clinical Trials** | Trial landscape by indication, phase, sponsor |

## How It Works

1. **Ask in natural language** — The agent parses your intent
2. **MCP-powered search** — Connects to PatSnap's pharmaceutical intelligence database
3. **Structured reports** — Generates professional-grade reports with pipeline tables, statistics, and insights

## Configuration

### API Key (Optional)

Set the `PATSNAP_API_KEY` environment variable in your Space settings to enable live pharmaceutical data. Without it, the demo uses a built-in knowledge base.

```bash
PATSNAP_API_KEY=your_key_here
```

## Powered By

- [PatSnap Life Sciences MCP](https://github.com/patsnap/skills)
- [Gradio](https://gradio.app)
