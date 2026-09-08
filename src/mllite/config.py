"""Typed configuration objects, loaded from YAML.

Every stage reads its settings from one of the nested dataclasses below, so a run
is fully described by a single config file plus the input data.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _subset(cls, payload: Optional[Dict[str, Any]]):
    """Build a dataclass from a dict, ignoring unknown keys but reporting them."""
    payload = dict(payload or {})
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config key(s): {unknown}")
    return cls(**payload)


@dataclass
class RunConfig:
    name: str = "run"
    output_dir: str = "outputs"
    seed: int = 42
    n_jobs: int = -1               # process-level parallelism for stages
    backend: str = "loky"          # joblib backend: loky | threading | sequential
    log_level: str = "INFO"
    gates: str = "open"
    # ^ open    : every gate measures and reports, but nothing is removed. Each candidate
    #             is carried through all four stages so you can watch where it stands.
    #             Costs more compute - every candidate gets a model.
    #   enforce : gates act. Failing features are dropped and never reach later stages,
    #             which is what makes the pipeline usable as an automatic filter.


@dataclass
class SplitConfig:
    mode: str = "random"           # random | column | time
    column: Optional[str] = None   # mode=column: col holding train/valid/test labels
    time_col: Optional[str] = None # mode=time: col to sort on
    valid_size: float = 0.2
    test_size: float = 0.2
    stratify: bool = False         # random mode, classification only


@dataclass
class DataConfig:
    path: str = ""                 # single table, split by `split` below
    paths: Dict[str, str] = field(default_factory=dict)
    # ^ already-split inputs: {train: ..., valid: ..., test: ...}. When set, this
    #   wins over `path` and no splitting happens - `split` is ignored entirely.
    format: str = "auto"           # auto | parquet | csv | pickle
    target: str = "y"
    weight_col: Optional[str] = None
    id_cols: List[str] = field(default_factory=list)
    missing_values: List[float] = field(default_factory=lambda: [-9999])
    nrows: Optional[int] = None    # cap rows read (csv only), for smoke runs
    split: SplitConfig = field(default_factory=SplitConfig)

    def __post_init__(self):
        if isinstance(self.split, dict):
            self.split = _subset(SplitConfig, self.split)


@dataclass
class FeatureConfig:
    """Which columns are the incumbent set and which are under evaluation."""
    base: List[str] = field(default_factory=list)
    new: List[str] = field(default_factory=list)
    base_prefix: Optional[str] = None   # infer base cols by prefix instead of listing
    new_prefix: Optional[str] = None
    exclude: List[str] = field(default_factory=list)


@dataclass
class DataQualityConfig:
    enabled: bool = False
    by_col: Optional[str] = None        # period column for stability (e.g. month)
    reference_period: Optional[str] = None   # period the others are compared against
    max_missing_rate: float = 0.99
    min_unique: int = 2
    max_psi: float = 0.25               # population stability index a feature may not exceed
    drop_failed: bool = False           # remove failing features before selection

    # Distribution consistency across the splits: is a variable shaped the same way
    # in valid/test as it is in train? Distinct from the period stability above,
    # which compares `by_col` periods and is still the unported AIME check.
    distribution_check: bool = True
    distribution_reference: str = "train"   # split the others are compared against
    distribution_bins: int = 10


@dataclass
class SpearmanConfig:
    enabled: bool = True
    target_min_abs: float = 0.0         # drop new feature if |rho| w/ target below
    redundancy_max_abs: float = 1.01    # drop if |rho| w/ a kept feature above


@dataclass
class MutualInfoConfig:
    enabled: bool = True
    bins: int = 20                      # quantile bins for the histogram estimator
    target_method: str = "histogram"    # histogram | sklearn
    target_min: float = 0.0             # min MI (nats) vs target
    redundancy_max: float = 1.01        # max normalized MI vs a kept feature
    redundancy_stat: str = "max"        # max | mean
    # ^ aggregates MI across many features for the *reported* redundancy column and the
    #   mrmr ranking. The drop gate always uses max: being a near-copy of one incumbent is
    #   what makes a candidate redundant, and a mean over a large pool would hide it.
    #   Use `mean` when ranking with mrmr against a large incumbent set.
    pairwise_against: str = "all"       # all | base | new


@dataclass
class FeatureSelectionConfig:
    enabled: bool = True
    sample_size: Optional[int] = 200_000  # rows sampled for correlation/MI
    on_split: str = "train"
    chunk_size: int = 25                  # new features per parallel task
    ranking: str = "spearman"             # spearman | mi | mrmr - the order candidates are
    # considered in. Order matters: the greedy redundancy screen keeps whichever member of a
    # correlated cluster it reaches first. `mrmr` re-scores after every pick, trading relevance
    # to the outcome against redundancy with what is already in the set.
    spearman: SpearmanConfig = field(default_factory=SpearmanConfig)
    mutual_info: MutualInfoConfig = field(default_factory=MutualInfoConfig)

    def __post_init__(self):
        if isinstance(self.spearman, dict):
            self.spearman = _subset(SpearmanConfig, self.spearman)
        if isinstance(self.mutual_info, dict):
            self.mutual_info = _subset(MutualInfoConfig, self.mutual_info)


@dataclass
class TuningConfig:
    """Random search over XGBoost hyperparameters, scored on the valid split.

    ``mode`` is the choice that matters for a champion/challenger comparison:

      shared        tune once on ``tune_on``, then give every variant the same
                    hyperparameters. The Gini difference between variants is then
                    attributable to the feature sets alone.
      per_variant   tune every variant separately. Each feature set gets its best
                    shot, which is fairer to a challenger whose optimum genuinely
                    differs - at the cost of mixing tuning luck into the gap.
    """
    enabled: bool = False
    mode: str = "shared"              # shared | per_variant
    tune_on: str = "base_plus_new"    # variant tuned when mode=shared
    n_trials: int = 20
    metric: str = "adj_gini"          # the only implemented objective; validated below
    seed: Optional[int] = None        # defaults to run.seed
    search_space: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    task: str = "regression"            # regression | binary
    params: Dict[str, Any] = field(default_factory=dict)
    num_boost_round: int = 500
    early_stopping_rounds: Optional[int] = 50
    verbose_eval: int = 0
    variants: List[str] = field(default_factory=lambda: ["base", "base_plus_new"])
    # variants: base | base_plus_new | new_only | leave_one_in | leave_one_out
    save_models: bool = True
    threads_per_model: Optional[int] = None  # None -> divide cores across variants
    tuning: TuningConfig = field(default_factory=TuningConfig)

    def __post_init__(self):
        if isinstance(self.tuning, dict):
            self.tuning = _subset(TuningConfig, self.tuning)


@dataclass
class ShapConfig:
    enabled: bool = True
    sample_size: int = 20_000
    on_split: str = "test"


@dataclass
class AnalysisConfig:
    metrics_on: List[str] = field(default_factory=lambda: ["train", "valid", "test"])
    capture_rate_percents: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])
    comparison_capture_percents: List[float] = field(default_factory=lambda: [0.05])
    # ^ which capture rates reach the comparison table and the report. Every percent in
    #   capture_rate_percents is still written to metrics_by_variant_split.csv; this
    #   only picks the headline one(s), since each adds a column per split.
    accuracy_bins: int = 10
    include_accuracy: bool = False
    # ^ calc_accuracy is a decile calibration measure, not classification accuracy, and
    #   it is dominated by sampling noise on small splits: with ~44 events a perfectly
    #   calibrated model scores ~0.75 +/- 0.11. Still written to
    #   metrics_by_variant_split.csv; this only controls the comparison table.
    baseline_variant: str = "base"      # variant that gini gain is measured against
    shap: ShapConfig = field(default_factory=ShapConfig)

    def __post_init__(self):
        if isinstance(self.shap, dict):
            self.shap = _subset(ShapConfig, self.shap)


@dataclass
class VerdictConfig:
    """Thresholds for the four gates a proposed feature has to clear."""
    enabled: bool = True
    gini_split: str = "test"            # split the gini gain is judged on
    min_gini_gain: float = 0.005        # below this, the feature did not move the model
    require_valid_too: bool = True      # the gain must hold on valid as well
    max_shap_rank_pct: float = 0.5      # must land in the top half by mean |SHAP|
    shap_variant: str = "base_plus_new"


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    verdict: VerdictConfig = field(default_factory=VerdictConfig)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Config":
        payload = copy.deepcopy(payload or {})
        cfg = cls(
            run=_subset(RunConfig, payload.pop("run", None)),
            data=_subset(DataConfig, payload.pop("data", None)),
            features=_subset(FeatureConfig, payload.pop("features", None)),
            data_quality=_subset(DataQualityConfig, payload.pop("data_quality", None)),
            feature_selection=_subset(FeatureSelectionConfig, payload.pop("feature_selection", None)),
            model=_subset(ModelConfig, payload.pop("model", None)),
            analysis=_subset(AnalysisConfig, payload.pop("analysis", None)),
            verdict=_subset(VerdictConfig, payload.pop("verdict", None)),
        )
        if payload:
            raise ValueError(f"unknown top-level config section(s): {sorted(payload)}")
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def gates_enforced(self) -> bool:
        return self.run.gates == "enforce"

    def validate(self) -> None:
        if self.data_quality.distribution_reference not in ("train", "valid", "test"):
            raise ValueError("data_quality.distribution_reference must be train|valid|test, "
                             f"got {self.data_quality.distribution_reference!r}")
        if self.data_quality.distribution_bins < 2:
            raise ValueError("data_quality.distribution_bins must be >= 2, "
                             f"got {self.data_quality.distribution_bins}")
        if self.run.gates not in ("open", "enforce"):
            raise ValueError(f"run.gates must be open|enforce, got {self.run.gates!r}")
        if self.model.task not in ("regression", "binary"):
            raise ValueError(f"model.task must be regression|binary, got {self.model.task!r}")
        if self.data.split.mode not in ("random", "column", "time"):
            raise ValueError(f"data.split.mode must be random|column|time, got {self.data.split.mode!r}")
        if self.data.paths:
            pass   # `split` is unused when the inputs arrive already split
        elif self.data.split.mode == "column" and not self.data.split.column:
            raise ValueError("data.split.mode=column requires data.split.column")
        elif self.data.split.mode == "time" and not self.data.split.time_col:
            raise ValueError("data.split.mode=time requires data.split.time_col")
        tuning = self.model.tuning
        if tuning.mode not in ("shared", "per_variant"):
            raise ValueError(f"model.tuning.mode must be shared|per_variant, got {tuning.mode!r}")
        if tuning.metric != "adj_gini":
            raise ValueError(
                "model.tuning.metric only supports 'adj_gini' - trials are scored with "
                f"calc_adj_gini on the valid split. Got {tuning.metric!r}, which would have "
                "been silently ignored.")
        if tuning.enabled and tuning.n_trials < 1:
            raise ValueError(f"model.tuning.n_trials must be >= 1, got {tuning.n_trials}")
        if tuning.enabled and tuning.mode == "shared" and tuning.tune_on not in self.model.variants:
            raise ValueError(
                f"model.tuning.tune_on={tuning.tune_on!r} is not among model.variants "
                f"{self.model.variants}; with leave_one_in/leave_one_out name a concrete "
                "variant such as 'base_plus_new'")
        if not 0 < self.verdict.max_shap_rank_pct <= 1:
            raise ValueError("verdict.max_shap_rank_pct must be in (0, 1], got "
                             f"{self.verdict.max_shap_rank_pct}")
        if self.feature_selection.ranking not in ("spearman", "mi", "mrmr"):
            raise ValueError("feature_selection.ranking must be spearman|mi|mrmr, "
                             f"got {self.feature_selection.ranking!r}")
        if self.feature_selection.mutual_info.redundancy_stat not in ("max", "mean"):
            raise ValueError("feature_selection.mutual_info.redundancy_stat must be max|mean, "
                             f"got {self.feature_selection.mutual_info.redundancy_stat!r}")
        if self.data.paths:
            unknown = sorted(set(self.data.paths) - {"train", "valid", "test"})
            if unknown:
                raise ValueError(f"data.paths keys must be train/valid/test, got extra: {unknown}")
            if "train" not in self.data.paths:
                raise ValueError("data.paths must include a 'train' entry")
        known_variants = {"base", "base_plus_new", "new_only", "leave_one_in", "leave_one_out"}
        bad = set(self.model.variants) - known_variants
        if bad:
            raise ValueError(f"unknown model.variants: {sorted(bad)}; allowed {sorted(known_variants)}")


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Read a YAML config, apply dotted-key overrides, and validate it."""
    with open(path, "r") as fh:
        payload = yaml.safe_load(fh) or {}
    for dotted, value in (overrides or {}).items():
        _set_dotted(payload, dotted, value)
    cfg = Config.from_dict(payload)
    cfg.validate()
    return cfg


def _set_dotted(payload: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = payload
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
