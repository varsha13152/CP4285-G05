# Data Layer

Shared data loading and preprocessing for KuaiRec.

## Usage

```python
from recsys.data import load_kuairec
from recsys.data.preprocessing import clip_watch_ratio, temporal_train_test_split

data = load_kuairec("data/raw")
print(data.small_matrix.shape)  # (4676570, 8)

# Preprocess
clean = clip_watch_ratio(data.small_matrix)
train, test = temporal_train_test_split(clean)
```

## Column Constants

Import from `recsys.data.schema` instead of hardcoding strings:

```python
from recsys.data.schema import USER_ID, VIDEO_ID, WATCH_RATIO
```
