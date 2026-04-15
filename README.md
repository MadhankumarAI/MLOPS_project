# Turing Tag

NER-aware machine translation. Named entities stay — everything else shifts.

The problem: translation APIs don't know that "Athmanandam" is a name, not two Hindi words. They'll happily translate it to "आत्मानंदम" (soul-happy). Turing Tag runs NER first, masks the entities, translates the rest, and stitches it back together. The name survives.

## What this does

```
Input: "Taj Mahal was built by Shah Jahan in Agra"
                ↓
        NER identifies: Taj Mahal (geo), Shah Jahan (per), Agra (geo)
                ↓
        Masked: "__ENT0__ was built by __ENT1__ in __ENT2__"
                ↓
        Translated: "__ENT0__ fue construido por __ENT1__ en __ENT2__"
                ↓
Output: "Taj Mahal fue construido por Shah Jahan en Agra"
```

Entities are color-coded in the UI, collected as they appear, and preserved across any target language.

## Architecture

```
pipeline/          NER model training (CRF, BiLSTM-CRF, BERT)
api/               FastAPI backend — NER + translation service
web/               React + TypeScript + Vite frontend
monitoring/        Prometheus config
tests/             pytest suite
```

**Pipeline** trains three NER models on the GMB corpus, tracks experiments with MLflow, and promotes the best one. **API** loads the winning model at startup and exposes REST + WebSocket endpoints. Translation is pluggable — swap Google Translate for MarianMT (local) via an env var. **Frontend** has request and realtime modes, highlights entities inline, and collects them in a sidebar.

## Models

| Model | Type | What it brings |
|---|---|---|
| CRF | Classical | Fast, interpretable, strong baseline |
| BiLSTM-CRF | Deep learning | Captures sequence context |
| BERT-NER | Transformer | Fine-tuned bert-base-uncased |

All three are evaluated on F1/precision/recall. The best by F1 gets registered and served.

## Setup

```bash
# clone and enter
git clone https://github.com/your-org/turing_tag.git
cd turing_tag

# backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# place the dataset
# download from: https://www.kaggle.com/datasets/namanj27/ner-dataset
mkdir -p data/raw
# move ner_dataset.csv into data/raw/

# preprocess
python -m pipeline.data.preprocess

# train (pick one or all)
python -m pipeline.training.train --model crf
python -m pipeline.training.train --model bilstm_crf
python -m pipeline.training.train --model bert_ner

# evaluate and promote best
python -m pipeline.training.evaluate

# run the api
uvicorn api.main:app --reload

# frontend (separate terminal)
cd web
npm install
npm run dev
```

## GPU

BiLSTM-CRF and BERT-NER use GPU automatically when available. If `torch.cuda.is_available()` returns `False`, you likely have the CPU-only PyTorch wheel:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## API

| Endpoint | Method | What |
|---|---|---|
| `/api/ner` | POST | Run NER on text, get tokens + tags + entities |
| `/api/translate` | POST | NER-aware translation |
| `/api/ws/translate` | WS | Realtime — streams NER + translation as you type |
| `/api/entities` | GET | All collected entities this session |
| `/api/entities` | DELETE | Clear collected entities |
| `/health` | GET | Liveness check |
| `/metrics` | GET | Prometheus metrics |

**Request body** for `/api/translate`:
```json
{ "text": "Shah Jahan built the Taj Mahal", "target_lang": "hi" }
```

**Response**:
```json
{
  "source_text": "Shah Jahan built the Taj Mahal",
  "translated_text": "Shah Jahan ने Taj Mahal बनवाया",
  "entities": [
    { "text": "Shah Jahan", "label": "per", "start": 0, "end": 10 },
    { "text": "Taj Mahal", "label": "geo", "start": 21, "end": 30 }
  ],
  "target_lang": "hi"
}
```

## Translation backends

Set `TRANSLATION_BACKEND` env var:

| Backend | Value | Needs |
|---|---|---|
| Google Translate | `google` (default) | `deep-translator` (included) |
| MarianMT | `marian` | Downloads Helsinki-NLP models locally |

Adding a new backend: implement `api/services/translation/base.py::Translator` and register it in `factory.py`.

## MLflow

Experiments log to a local SQLite database. To view:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Opens a dashboard at `localhost:5000` where you can compare runs.

## DVC

`dvc.yaml` defines the full reproducible pipeline: preprocess → train × 3 → evaluate. Run it end-to-end with:

```bash
dvc repro
```

## Tests

```bash
pytest tests/ -v
```

Covers: data loading, vocab construction, BiLSTM-CRF forward/backward, CRF feature extraction, entity masking/restoration, API endpoints.

## CI

GitHub Actions runs on every push and PR to `main`:

- **Lint** — flake8 across pipeline, api, tests
- **Pipeline tests** — data processing, model shapes, feature extraction
- **API tests** — endpoint health, translation service logic
- **Frontend** — TypeScript type-check + Vite production build
- **Train** (main only) — preprocess → train CRF → evaluate → upload artifacts

## Project structure

```
turing_tag/
├── .github/workflows/ci.yml
├── pipeline/
│   ├── config.py
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── crf_model.py
│   │   ├── bilstm_crf.py
│   │   └── bert_ner.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── registry/
│       └── promote.py
├── api/
│   ├── main.py
│   ├── deps.py
│   ├── routes/
│   │   ├── ner.py
│   │   ├── translate.py
│   │   └── stream.py
│   ├── services/
│   │   ├── ner_service.py
│   │   ├── translate_service.py
│   │   └── translation/
│   │       ├── base.py
│   │       ├── factory.py
│   │       ├── google.py
│   │       └── marian.py
│   └── schemas/
│       └── models.py
├── web/                         React + TS + Vite
├── monitoring/prometheus.yml
├── tests/
├── params.yaml
├── dvc.yaml
└── requirements.txt
```

## License

MIT
