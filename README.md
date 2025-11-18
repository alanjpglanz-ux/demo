# Autarkeia Metrics Checker
**Internal Stability & Emergence Diagnostics for Machine Learning Systems**  
**Patent Pending – US Provisional 63/815,941**

Autarkeia measures **internal stabilization**, **novelty collapse**, and **emergent structure** inside ML models.  
It introduces a new mathematical framework — **HOEC (Higher-Order Emergent Calculus)** — and an order parameter for internal equilibrium (**Completion**).

## Why This Matters
Loss/accuracy measure external behavior. Autarkeia measures the **internal geometry** of learning:
- Detects **phase transitions** and **emergent behavior**
- Flags **false convergence** and **internal turbulence**
- Provides a **stability verdict** independent of loss

## Key Metrics
- **Ī (novelty-rate)** via reconstruction residuals
- **Completion** (order parameter) combining normalized novelty, entropy, subspace drift, and flip-rate
- **HOEC**: Δ/Δ², curvature, hysteresis, **EOI (Emergence Order Index)**

## Architecture
1. Representation Builder (PPMI → SVD latent subspace)
2. Novelty Measurement (normalized reconstruction residual)
3. Self-Description Feedback (anchor promotion + reweighting)
4. Completion (convex combination of internal stability signals)
5. HOEC (derivatives, curvature, hysteresis, EOI)
6. Dashboard + REST API

## Quick Start (Dashboard)
```bash
pip install -r requirements.txt
python demo_app/app.py
# open http://localhost:8008
```

## Quick Start (Python)
```python
from autarkeia.engine.engine_core import AutarkeiaMetrics

# toy embeddings: T time-steps of N vectors in D-dim space
series = [np.random.randn(512, 128) for _ in range(6)]
m = AutarkeiaMetrics.from_embeddings(series)
m.run(max_iters=5)
print(m.summary())
```

## Repo Layout
```
autarkeia/engine/     # core API (skeleton)
demo_app/             # Flask dashboard
examples/             # 3 notebooks (skeletons)
docs/                 # PDFs + diagrams
tests/                # minimal tests
Dockerfile
docker-compose.yml
run.sh
setup.py
requirements.txt
LICENSE
README.md
```

## Status
Acquisition-ready skeleton for technical due diligence.  
For initial evaluation artifacts (1-pagers, executive summary), use the **Light Package**.
