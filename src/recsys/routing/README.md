# Intent Routing

Owned by **Varsha**. Maps an `IntentSignal` (from intent understanding) to a `RetrievalRequest` that tells the retrieval stage what strategies to use and with what weights.

## How it works

Each `IntentType` maps to a list of `(RetrievalStrategy, weight)` pairs in `strategy_map.py`. The router looks up the mapping and builds a `RetrievalRequest`.

Example: `BINGE_WATCH` → 50% collaborative, 40% content-based, 10% popularity.

## Usage

```python
from recsys.routing import IntentRouter

router = IntentRouter()
request = router.route(intent_signal)

# request.strategies = [
#   StrategyWeight(COLLABORATIVE, 0.5),
#   StrategyWeight(CONTENT_BASED, 0.4),
#   StrategyWeight(POPULARITY, 0.1),
# ]
```
