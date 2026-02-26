# v7 Proposal — Temporal Shift & Overfitting Improvements

Standalone proposal for teammates. See [EXPERIMENTS.md](EXPERIMENTS.md) for full experiment log.

---

## 1. Context — Why v7

| Metric | Value |
|--------|-------|
| November holdout AUC (v6) | ~0.63 |
| Kaggle public AUC | ~0.60 |
| Gap | ~0.03 |

**Problem:** The model overfits to November. Patterns that score well on our Nov holdout do not generalize to December (Kaggle test set). Temporal shift between Nov and Dec is hurting leaderboard performance.

**Root causes:**
- Validation uses Nov as the primary metric; config selection optimizes for Nov, not for generalization.
- Some features may be Nov-specific (seasonality, year-end effects).
- Entity rates are computed without recency weighting — recent behavior may matter more for Dec predictions.

---

## 2. Proposed Validation Protocol

**Current:** 3-fold temporal CV; config selection by Nov AUC (Fold 3).

**Proposed:** 4-fold forward validation — select configs by **mean AUC across all folds**, not Nov-only.

| Fold | Train months | Val month |
|------|--------------|-----------|
| 1 | Jan–Jul | Aug |
| 2 | Jan–Aug | Sep |
| 3 | Jan–Sep | Oct |
| 4 | Jan–Oct | Nov |

**Metric:** Mean AUC (and std) across the 4 folds. Use mean for config selection; report std for stability.

**Rationale:** Reduces overfitting to November by forcing the model to perform well on Aug, Sep, Oct as well. A config that generalizes across months should also generalize to December.

---

## 3. Proposed Temporal Features

| Feature group | Description |
|---------------|-------------|
| **Recency-weighted entity rates** | Weight recent FPD events more than older ones when computing entity FPD rates. Captures changing behavior over time. |
| **Seasonal features** | `quarter` (1–4), `is_year_end` (Nov/Dec), `month_sin`/`month_cos` for smooth seasonality. |
| **Payment recency** | Distinguish recent zero-recovery vs overall zero-recovery in payment portfolio. Recent behavior may be more predictive for Dec. |

---

## 4. Implementation Tasks

| # | Task | Suggested owner |
|---|------|-----------------|
| 1 | Update `CV_SPLITS` in pipeline to 4-fold forward (Aug, Sep, Oct, Nov) | — |
| 2 | Add recency-weighted entity rate option (exponential decay or sliding window) | — |
| 3 | Add seasonal features (quarter, is_year_end, month encoding) | — |
| 4 | Add payment recency features (recent vs overall zero-recovery) | — |
| 5 | Update `_run_v6_experiments.py` to use new validation; re-run config sweep | — |

---

## 5. Success Criteria

- **Mean 4-fold AUC** maintained or improved vs v6 (Nov AUC 0.63408).
- **Kaggle public AUC** target: ~0.62 (closing the Nov–Kaggle gap).
- **Stability:** Std of 4-fold AUC should be reasonable; high variance suggests unstable config.

---

## 6. References

- [EXPERIMENTS.md](EXPERIMENTS.md) — Full experiment log, scorecard
- [v6_experiments_results.csv](v6_experiments_results.csv) — v6 config sweep results
- [run_v6.log](run_v6.log) — v6 full run log
- [RESULTS.md](RESULTS.md) — v5/v6 results summary
