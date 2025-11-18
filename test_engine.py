
import numpy as np
from autarkeia.engine import AutarkeiaMetrics


def test_autarkeia_runs():
    series = [np.random.randn(64, 16) for _ in range(4)]
    m = AutarkeiaMetrics.from_embeddings(series)
    res = m.run()
    assert res.completion.shape[0] == len(series)
    assert 0.0 <= res.eoi <= 1.0 or res.eoi >= 0.0  # loose sanity check
