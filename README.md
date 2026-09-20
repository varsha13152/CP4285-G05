# KuaiRec Recommendation System

A modern recommendation pipeline built on the [KuaiRec dataset](https://github.com/chongminggao/KuaiRec), a fully-observed dataset from the Kuaishou video-sharing app.

## Architecture

```
User → [Intent Understanding] → [Intent Routing] → [Candidate Retrieval] → [Ranking] → Recommendations
```

| Stage | Owner | Package | Input | Output |
|-------|-------|---------|-------|--------|
| Intent Understanding | Varsha | `src/recsys/intent/` | User ID + behavior data | `IntentSignal` |
| Intent Routing | Varsha | `src/recsys/routing/` | `IntentSignal` | `RetrievalRequest` |
| Candidate Retrieval | TBD | `src/recsys/retrieval/` | `RetrievalRequest` | `CandidateSet` |
| Ranking | TBD | `src/recsys/ranking/` | `CandidateSet` | `RankedList` |

All inter-stage types are defined in `src/recsys/contracts/` — import from there.

## Setup

```bash
# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"

# Download KuaiRec data
python scripts/download_data.py

# Verify
python -c "from recsys.data.loader import load_kuairec; d = load_kuairec(); print(d.small_matrix.shape)"
```

## Data

Place KuaiRec CSV files in `data/raw/`. The dataset includes:

- `small_matrix.csv` — 1,411 users × 3,327 videos, 99.6% dense interaction matrix
- `big_matrix.csv` — 7,176 users × 10,728 videos, 16.3% dense
- `user_features.csv` / `user_features_raw.csv` — User profiles and demographics
- `item_categories.csv` — Video category tags
- `item_daily_features.csv` — Daily video engagement metrics
- `kuairec_caption_category.csv` — Video captions and hierarchical categories
- `social_network.csv` — User friend lists

## Running

```bash
# Run tests
python -m pytest tests/ -v

# Run pipeline (uses mocks for unfinished stages)
python scripts/run_pipeline.py --user_id 42
```

## Development

Each team member works on their own package. See the README inside each module for interface contracts.

```bash
# Work on your module
git checkout -b feat/<your-module>

# Run your tests
python -m pytest tests/test_<module>.py -v

# When done, open a PR to main
```
