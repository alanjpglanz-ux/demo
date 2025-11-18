
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class AutarkeiaResult:
    novelty: np.ndarray
    entropy: np.ndarray
    subspace_drift: np.ndarray
    flip_rate: np.ndarray
    completion: np.ndarray
    d_completion: np.ndarray
    d2_completion: np.ndarray
    curvature: np.ndarray
    hysteresis: float
    eoi: float
    meta: Dict[str, Any] = field(default_factory=dict)


class AutarkeiaMetrics:
    """
    Core Autarkeia engine.

    Given a time-series of embedding matrices X_t in R^{N x D}, this class computes:
      - novelty-rate Ī_t via reconstruction residuals
      - entropy of the singular value spectrum
      - subspace drift via principal angles between PCA subspaces
      - flip-rate via sign / threshold flips in cosine similarity over time
      - a unified Completion order parameter
      - HOEC-style higher-order signals: derivatives, curvature, hysteresis, EOI
    """

    def __init__(self, series: List[np.ndarray], rank: int = 64, completion_weights=None):
        if len(series) < 2:
            raise ValueError("AutarkeiaMetrics requires at least 2 time steps of embeddings.")

        self.series = [self._to_2d(x) for x in series]
        self.T = len(self.series)
        self.rank = rank
        self.completion_weights = completion_weights or {
            "novelty": 0.35,
            "entropy": 0.2,
            "subspace_drift": 0.25,
            "flip_rate": 0.2,
        }
        if not np.isclose(sum(self.completion_weights.values()), 1.0):
            raise ValueError("Completion weights must sum to 1.0")

        self._result: Optional[AutarkeiaResult] = None

    @classmethod
    def from_embeddings(cls, series: List[np.ndarray], **kwargs) -> "AutarkeiaMetrics":
        return cls(series, **kwargs)

    @staticmethod
    def _to_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        return x.astype(float)

    def run(self, max_iters: int = 1) -> AutarkeiaResult:
        """
        Main entry point.
        max_iters is kept for API compatibility; the current implementation
        is single-pass over the time series.
        """
        novelty = self._compute_novelty()
        entropy = self._compute_entropy()
        subspace_drift = self._compute_subspace_drift()
        flip_rate = self._compute_flip_rate()

        completion = self._compute_completion(
            novelty=novelty,
            entropy=entropy,
            subspace_drift=subspace_drift,
            flip_rate=flip_rate,
        )

        d_completion, d2_completion, curvature = self._compute_derivatives(completion)
        hysteresis = self._compute_hysteresis(completion)
        eoi = self._compute_eoi(completion, novelty, entropy, subspace_drift, flip_rate)

        self._result = AutarkeiaResult(
            novelty=novelty,
            entropy=entropy,
            subspace_drift=subspace_drift,
            flip_rate=flip_rate,
            completion=completion,
            d_completion=d_completion,
            d2_completion=d2_completion,
            curvature=curvature,
            hysteresis=hysteresis,
            eoi=eoi,
            meta={
                "rank": self.rank,
                "T": self.T,
                "completion_weights": self.completion_weights,
            },
        )
        return self._result

    # ----------------- Core metrics -----------------

    def _compute_novelty(self) -> np.ndarray:
        """
        Novelty-rate Ī_t: normalized reconstruction residual when projecting X_t
        into the PCA subspace learned from X_{t-1}.
        """
        novelty = np.zeros(self.T)
        for t in range(1, self.T):
            X_prev = self.series[t - 1]
            X = self.series[t]

            # PCA via SVD on previous step
            U, S, Vt = np.linalg.svd(
                X_prev - X_prev.mean(axis=0, keepdims=True), full_matrices=False
            )
            k = min(self.rank, Vt.shape[0])
            basis = Vt[:k].T  # D x k

            X_centered = X - X.mean(axis=0, keepdims=True)
            proj = X_centered @ basis @ basis.T
            residual = X_centered - proj

            num = np.linalg.norm(residual, "fro")
            den = np.linalg.norm(X_centered, "fro") + 1e-8
            novelty[t] = num / den

        novelty[0] = novelty[1]  # copy first non-zero value
        return novelty

    def _compute_entropy(self) -> np.ndarray:
        """
        Spectral entropy of singular values at each time step.
        """
        ent = np.zeros[self.T]
        for t, X in enumerate(self.series):
            Xc = X - X.mean(axis=0, keepdims=True)
            _, S, _ = np.linalg.svd(Xc, full_matrices=False)
            p = S / (S.sum() + 1e-8)
            p = p[p > 0]
            ent[t] = -(p * np.log(p + 1e-12)).sum()
        return ent

    def _compute_subspace_drift(self) -> np.ndarray:
        """
        Subspace drift via principal angle surrogate between PCA subspaces
        at t-1 and t.
        """
        drift = np.zeros(self.T)
        prev_basis = None
        for t, X in enumerate(self.series):
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            k = min(self.rank, Vt.shape[0])
            basis = Vt[:k].T  # D x k

            if prev_basis is not None:
                # cosines of principal angles via singular values of B_prev^T B
                M = prev_basis.T @ basis
                _, S_m, _ = np.linalg.svd(M, full_matrices=False)
                cos_thetas = np.clip(S_m, 0.0, 1.0)
                thetas = np.arccos(cos_thetas)
                drift[t] = float(np.mean(thetas))
            else:
                drift[t] = 0.0

            prev_basis = basis
        return drift

    def _compute_flip_rate(self, threshold: float = 0.3) -> np.ndarray:
        """
        Flip-rate: fraction of vectors whose cosine similarity to their
        previous-time counterpart crosses below a threshold or changes sign.
        """
        flips = np.zeros(self.T)
        for t in range(1, self.T):
            X_prev = self.series[t - 1]
            X = self.series[t]
            n = min(X_prev.shape[0], X.shape[0])
            X_prev = X_prev[:n]
            X = X[:n]

            # cosine similarity row-wise
            num = np.sum(X_prev * X, axis=1)
            den = (
                np.linalg.norm(X_prev, axis=1) * np.linalg.norm(X, axis=1) + 1e-8
            )
            cos = num / den

            flip_mask = (cos < threshold).astype(float)
            flips[t] = flip_mask.mean()

        flips[0] = flips[1]
        return flips

    # ----------------- Aggregation & HOEC -----------------

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-8:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn + 1e-8)

    def _compute_completion(
        self,
        novelty: np.ndarray,
        entropy: np.ndarray,
        subspace_drift: np.ndarray,
        flip_rate: np.ndarray,
    ) -> np.ndarray:
        # normalize all signals
        n = self._normalize(novelty)
        e = self._normalize(entropy)
        d = self._normalize(subspace_drift)
        f = self._normalize(flip_rate)

        # "completion" is high when novelty, drift, flips are low and
        # spectral entropy is moderate–low (more ordered)
        completion = (
            self.completion_weights["novelty"] * (1.0 - n)
            + self.completion_weights["entropy"] * (1.0 - e)
            + self.completion_weights["subspace_drift"] * (1.0 - d)
            + self.completion_weights["flip_rate"] * (1.0 - f)
        )
        return completion

    def _compute_derivatives(self, completion: np.ndarray):
        d1 = np.zeros_like(completion)
        d2 = np.zeros_like(completion)

        for t in range(1, self.T):
            d1[t] = completion[t] - completion[t - 1]
        for t in range(2, self.T):
            d2[t] = d1[t] - d1[t - 1]

        curvature = np.abs(d2)
        return d1, d2, curvature

    def _compute_hysteresis(self, completion: np.ndarray) -> float:
        """
        Simple hysteresis proxy: compare forward vs. backward completion paths.
        """
        fwd = completion
        bwd = completion[::-1]
        return float(np.mean(np.abs(fwd - bwd)))

    def _compute_eoi(
        self,
        completion: np.ndarray,
        novelty: np.ndarray,
        entropy: np.ndarray,
        subspace_drift: np.ndarray,
        flip_rate: np.ndarray,
    ) -> float:
        """
        Emergence Order Index (EOI): a scalar summary that rewards
        high completion and penalizes volatility / turbulence.
        """
        c_norm = self._normalize(completion)
        n_norm = self._normalize(novelty)
        d_norm = self._normalize(subspace_drift)
        f_norm = self._normalize(flip_rate)

        stability = float(c_norm.mean())
        turbulence = float(0.5 * n_norm.mean() + 0.3 * d_norm.mean() + 0.2 * f_norm.mean())

        eoi = max(0.0, stability - turbulence)
        return eoi

    # ----------------- Convenience -----------------

    @property
    def result(self) -> AutarkeiaResult:
        if self._result is None:
            raise RuntimeError("run() must be called before accessing result.")
        return self._result

    def summary(self) -> str:
        if self._result is None:
            raise RuntimeError("run() must be called before summary().")

        r = self._result
        lines = []
        lines.append("Autarkeia Metrics Summary")
        lines.append("=" * 32)
        lines.append(f"Time steps: {self.T}")
        lines.append(f"Rank: {r.meta.get('rank')}")
        lines.append("")
        lines.append(f"Completion: mean={r.completion.mean():.3f}, std={r.completion.std():.3f}")
        lines.append(f"Novelty-rate Ī: mean={r.novelty.mean():.3f}")
        lines.append(f"Subspace drift: mean={r.subspace_drift.mean():.3f}")
        lines.append(f"Flip-rate: mean={r.flip_rate.mean():.3f}")
        lines.append("")
        lines.append(f"HOEC: hysteresis={r.hysteresis:.3f}, curvature_mean={r.curvature.mean():.3f}")
        lines.append(f"EOI (Emergence Order Index): {r.eoi:.3f}")
        return "\n".join(lines)

if __name__ == "__main__":
    # quick self-test with random data
    series = [np.random.randn(512, 128) for _ in range(6)]
    m = AutarkeiaMetrics.from_embeddings(series)
    res = m.run()
    print(m.summary())
