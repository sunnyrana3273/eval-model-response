# notebooks/

Exploratory analysis goes here. Notebooks should only **read** experiment artefacts and should never
be required by the pipeline, which runs entirely through the `afo` CLI.

Starting point:

```python
from afo.collection.storage import ExperimentStore
from afo.monitors.training import load_dataset

store = ExperimentStore("experiments/main-simulated")
ds, trajs = load_dataset(store)          # step-level features, activations, labels, metadata
violations = store.load_trajectories(environment="prompt_injection")
```

Planned analyses (see `docs/limitations.md`):

- post-exposure-only evaluation;
- grouped cross-fitting over all task families;
- error analysis of behavioural false alarms.
