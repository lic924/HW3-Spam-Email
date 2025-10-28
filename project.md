# Project Context

## Purpose
本專案基於《Hands-On Artificial Intelligence for Cybersecurity》第 3 章「Spam Email」案例，  
目標是建立垃圾郵件（Spam）與正常郵件（Ham）的分類器，並以 Streamlit 呈現資料分佈、特徵權重、模型評估與即時推論結果。

## Assumptions (explicit)
- Repository may contain multiple language subprojects (Node.js + TypeScript scaffold and Python ML experiments). Confirm preferred stacks before major changes.
- Package managers: npm for Node, pip/venv or poetry for Python depending on user's preference.
- Source layout:
  - Node: `src/` for TS sources, `test/` for tests
  - Python: `src/` or `src/ml/` for ML code, `data/` for datasets, `models/` for artifacts, `notebooks/` for exploration
- CI: none configured yet; prefer GitHub Actions for CI when added

If these assumptions are incorrect, the agent must ask one focused question before scaffolding.

## Tech Stack
- **語言**：Python 3.11+
- **主要套件**：
  - streamlit == 1.50.0
  - pandas == 2.3.3
  - scikit-learn == 1.7.2
  - numpy == 2.3.4
  - matplotlib, plotly, joblib, scipy
- **環境管理**：pip + venv
- **部署平台**：Streamlit Cloud

## Project Conventions

### Code Style
- Follow language-idiomatic linters/formatters:
  - Node/TS: ESLint + Prettier
  - Python: black + ruff/flake8
- File naming: kebab-case for filenames, PascalCase for exported classes, snake_case for Python modules, camelCase for JS/TS functions and variables.

### Architecture Patterns
- Keep capabilities single-purpose and isolated under directories or packages.
- For ML work, keep data, models, and notebooks separated: `data/`, `models/`, `notebooks/`.

### Testing Strategy
- Node: Jest
- Python: pytest
- Small, fast unit tests for CI; separate integration tests for longer-running tasks.

### Git Workflow
- Feature branches: `feature/<short-desc>` or `add-<thing>` for change proposals.
- Use PRs for review. Reference `openspec/changes/<change-id>/proposal.md` in PR description.

## Domain Context
- Spam/sms classification baseline is the first ML capability (see `openspec/changes/add-spam-classification-baseline`).

## Important Constraints
- Do NOT commit raw datasets, model artifacts, or large notebooks. Use `data/` with `.gitignore` entries and provide scripts to fetch canonical datasets.
- If adding credentials or external integrations, store connection details in secure secrets (do not commit).

## External Dependencies
- None by default. When present, document them in the relevant `openspec/changes/*/proposal.md` and add a `requirements.txt` or `package.json` in the subproject.

## How agents should use this file
- Read and confirm assumptions before making irreversible changes.
- Document any added subproject (language, package manager, key commands) in `README.md` at repo root.
- If scaffolding is requested, create a change proposal under `openspec/changes/` and add `proposal.md`, `tasks.md`, and spec deltas.
- Run `openspec validate <change-id> --strict` before requesting approval.

---
## Project: Spam Email Classifier

### Overview
- Dataset: `datasets/sms_spam_no_header.csv`
- Goal: Predict whether a given SMS is spam or ham
- Model: Logistic Regression with TF-IDF features
- UI: Streamlit app that provides:
  - Class distribution & token statistics
  - Model performance (ROC, PR, F1)
  - Threshold tuning (precision/recall sweep)
  - Live inference text box

### Deliverables
1. GitHub repo with code and documentation  
2. Online demo (Streamlit): [https://lic924-hw3-spam-email-app-streamlit-tkfzpe.streamlit.app/](https://lic924-hw3-spam-email-app-streamlit-tkfzpe.streamlit.app/)  
3. Tutorial video (YouTube link after upload)  
4. OpenSpec documentation (project.md + AGENTS.md)

### Folder Structure
```bash
Chapter03/
├── datasets/
├── models/
├── sources/
├── app_streamlit.py
├── prepare_dataset.py
├── requirements.txt
└── README.md
```
Last-updated-by-agent: merged-spam-ml
