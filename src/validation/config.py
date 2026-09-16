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
class MissingIndicatorConfig:
    """Add a 0/1 column recording where a feature was missing.

    Worth enabling when missingness is itself informative rather than incidental -
    a ratio undefined because its denominator is zero, a field only populated for
    one product. The value is unknown either way, but *that* it is unknown can
    carry signal, and a NaN alone throws that away.

    Indicators are derived after sentinels and infinities have been converted, so
    they capture every route to missing rather than only literal nulls.
    """
    enabled: bool = False
    scope: str = "candidates"        # candidates | all
    min_missing_rate: float = 0.01   # skip columns barely ever missing
    suffix: str = "_is_missing"
    treat_as: str = "new"
    # ^ new    : indicators are candidates in their own right, which is usually
    #            right - the incumbent model did not carry them either.
    #   source : an indicator joins whichever list its source column is in, for
    #            when the incumbent model already had these flags.


@dataclass
class DataConfig:
    paths: Dict[str, str] = field(default_factory=dict)
    # ^ {train: ..., valid: ..., test: ...} - the split, made once by the dataset's
    #   prepare step and read exactly as written. The pipeline never re-splits.
    format: str = "auto"           # auto | parquet | csv | pickle
    target: str = "y"
    weight_col: Optional[str] = None
    id_cols: List[str] = field(default_factory=list)
    missing_values: List[float] = field(default_factory=lambda: [-9999])
    missing_indicators: MissingIndicatorConfig = field(default_factory=MissingIndicatorConfig)

    def __post_init__(self):
        if isinstance(self.missing_indicators, dict):
            self.missing_indicators = _subset(MissingIndicatorConfig, self.missing_indicators)


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
    """Thresholds for the four gates a proposed feature has to clear.

    Test is the hold-out and the only evidence about a proposed feature. Valid
    is a tool: it tunes hyperparameters, stops training, and gives the discovery
    screen its feedback. That division is what makes the test number mean
    something, and it decides the default below.
    """
    enabled: bool = True
    gini_split: str = "test"            # split the gini gain is judged on
    min_gini_gain: float = 0.005        # below this, the feature did not move the model
    # Off, because valid is machinery rather than evidence. A discovered feature
    # was *selected* on valid rows, so requiring a valid gain asks it to repeat
    # the thing it was picked for - a gate that cannot fail for the reason it
    # exists. It stays meaningful for hand-declared candidates, which discovery
    # never touched, so one flag would mean two different things depending on
    # where the candidate came from. Set it True only for a run with no
    # discovery, where valid really is an independent second look.
    require_valid_too: bool = False
    max_shap_rank_pct: float = 0.5      # must land in the top half by mean |SHAP|
    shap_variant: str = "base_plus_new"


@dataclass
class LLMConfig:
    """Which model discovery asks, and how patient to be with it.

    Model choice is configuration, not environment: it belongs in the run's YAML
    where git tracks it, so a finished run says what produced it. Only the
    credential lives in .env.
    """
    backend: str = "openai"           # openai | safechain
    model: str = "gpt-4o-mini"        # an openai name, or a safechain key
    reasoning_effort: Optional[str] = None
    # ^ low | medium | high, or None. ONLY reasoning models accept it; gpt-4o
    #   rejects it outright with a 400, so it must stay None alongside one.
    system_prompt: str = ""
    # Calls stall rather than slow down: a call still running at stall_retry_s is
    # re-issued, and timeout_s stays generous enough that the retry can outlast a
    # stall that does resolve. See discovery/llm.py.
    timeout_s: float = 180.0
    stall_retry_s: float = 40.0
    max_attempts: int = 3
    backoff_s: float = 5.0


@dataclass
class DiscoveryConfig:
    """Propose candidate features, then screen them on a small sample.

    The screen is deliberately cheap and approximate: a generation loop needs a
    signal every round, while the real decision is made later by leave_one_in
    and the verdict gates over the full splits.
    """
    enabled: bool = False
    strategy: str = "caafe"           # caafe | elfgym | ferg | featllm | promptfe
    task_description: str = ""        # what the dataset is, in domain terms
    # Domain background for the task - how the score is used, what the incumbent
    # model leans on, which quantities must not be mixed. Fed to the proposer as
    # its own block. Inline for a line or two; a path when it is a page, or when
    # it must stay out of the repository.
    task_context: str = ""
    task_context_path: Optional[str] = None

    # Each round asks for a batch and screens it one feature at a time, folding
    # every outcome into the history the next round sees.
    batch_size: int = 4

    # --- stopping rule ----------------------------------------------------
    # max_rounds always applies; the rest end a run early when continuing cannot
    # help. One round by default: propose a batch, screen it, stop. Raise it to
    # let the proposer react to its own feedback.
    max_rounds: int = 1
    target_features: Optional[int] = None   # stop once this many are worth keeping
    patience: Optional[int] = None          # stop after N rounds that kept nothing
    max_candidates: Optional[int] = None    # hard cap on proposals screened

    # Which metric the screen scores with. Kept separate from the analysis
    # metrics on purpose: the screen needs one cheap number for feedback, while
    # analysis reports the whole bundle.
    metric: str = "adj_gini"          # adj_gini | capture_rate
    capture_percent: float = 0.05     # used when metric is capture_rate

    # What the screen forwards to the expensive stages.
    #
    # None (the default) forwards every candidate that RAN, filtering only the
    # broken ones. That is deliberate: the screen's delta comes from one small
    # held-out sample, so for a feature a boosted model can already approximate
    # it is noise-dominated and lands negative about half the time. Filtering on
    # it would discard good features by coin flip before leave_one_in and the
    # verdict gates - the stages that exist to decide - ever saw them.
    #
    # Set a number to pre-filter anyway, e.g. to cap how many candidates reach
    # the expensive stages on a large batch.
    min_delta: Optional[float] = None
    # The rows the screen fits each proposal on and scores it on:
    #   splits : fit on the train split, score on the valid split, as prepared
    #   files  : fit and score on samples the prepare step drew from train and
    #            valid, in the same format as the splits, at screen_paths
    #            {train: ..., valid: ...} - e.g. class-balanced when positives
    #            are rare
    # Scoring on valid rows means the proposer's feedback comes from valid, so
    # valid stops being an independent check on what it proposes; test still is.
    # To keep valid independent, sample both files from disjoint rows of train.
    screen_data: str = "splits"
    screen_paths: Dict[str, str] = field(default_factory=dict)
    screen_boost_rounds: int = 200
    # What the columns mean. Without this a proposer sees "X36" and can only
    # guess; with it, real-world knowledge becomes usable, which is the whole
    # premise of an LLM proposing features at all. Either inline, or a JSON file
    # of {column: description}.
    column_descriptions: Dict[str, str] = field(default_factory=dict)
    column_descriptions_path: Optional[str] = None
    # Which columns are coded categories rather than measurements, so the prompt
    # shows their levels instead of a meaningless range. Declare whichever list
    # is shorter: a clinical table is mostly coded categories with a dozen real
    # measurements, so naming the 12 continuous ones beats naming 99 categorical.
    categorical_columns: List[str] = field(default_factory=list)
    # None means "not declared"; an empty list means "no column is continuous",
    # which is a real declaration - a permissions table is 86 binary flags and
    # every one of them must be shown as levels rather than a range.
    continuous_columns: Optional[List[str]] = None

    # The example rows shown to the proposer: a CSV in the same format as the
    # splits plus a `batch` column, written by the dataset's prepare step and
    # clustered per class on train so the rows cover the table and every class
    # appears (see preprocessing/shots.py).
    # Round r shows batch r, so successive rounds see different rows.
    few_shot_path: Optional[str] = None
    # A running record of every proposal across discovery runs: its code, the
    # screen's score, and the final verdict with the reason - accepted as a
    # candidate, or rejected at which gate and why. Each run reads it into the
    # prompt, so an idea already judged is not proposed again, and appends its
    # own proposals once the verdict is in. Round numbers continue from it, and
    # so does the rotation through few_shot_path's batches. Accepted features
    # stay candidates: the incumbent set never changes. Unset, a run starts with
    # no memory; delete the file to start a fresh study.
    history_path: Optional[str] = None
    # A candidate whose largest magnitude exceeds this multiple of its own 99th
    # percentile is rejected before it can reach a model. See discovery/guards.py.
    spike_factor: float = 1000.0
    # Screen-level redundancy limit. None inherits
    # feature_selection.spearman.redundancy_max_abs, so the screen enforces the
    # rule the gate will apply rather than a second, divergent one.
    redundancy_max_abs: Optional[float] = None
    llm: LLMConfig = field(default_factory=LLMConfig)

    def __post_init__(self):
        if isinstance(self.llm, dict):
            self.llm = _subset(LLMConfig, self.llm)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
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
            discovery=_subset(DiscoveryConfig, payload.pop("discovery", None)),
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
        if self.discovery.enabled:
            if self.discovery.llm.backend not in ("openai", "safechain"):
                raise ValueError("discovery.llm.backend must be openai|safechain, "
                                 f"got {self.discovery.llm.backend!r}")
            if self.discovery.screen_data not in ("splits", "files"):
                raise ValueError("discovery.screen_data must be splits|files, "
                                 f"got {self.discovery.screen_data!r}")
            if (self.discovery.screen_data == "files"
                    and sorted(self.discovery.screen_paths) != ["train", "valid"]):
                raise ValueError("discovery.screen_data=files needs discovery.screen_paths "
                                 "with a train and a valid entry")
            if self.discovery.max_rounds < 1:
                raise ValueError("discovery.max_rounds must be >= 1, got "
                                 f"{self.discovery.max_rounds}")
            if self.discovery.batch_size < 1:
                raise ValueError("discovery.batch_size must be >= 1, got "
                                 f"{self.discovery.batch_size}")
            known = ("caafe", "elfgym", "ferg", "featllm", "promptfe")
            if self.discovery.strategy not in known:
                raise ValueError(f"discovery.strategy must be one of {known}, got "
                                 f"{self.discovery.strategy!r}")
            if self.discovery.metric not in ("adj_gini", "capture_rate"):
                raise ValueError("discovery.metric must be adj_gini|capture_rate, got "
                                 f"{self.discovery.metric!r}")
            if self.discovery.categorical_columns and self.discovery.continuous_columns is not None:
                raise ValueError(
                    "set either discovery.categorical_columns or "
                    "discovery.continuous_columns, not both"
                )
            if not self.discovery.task_description.strip():
                raise ValueError("discovery.task_description is required when discovery is "
                                 "enabled: the proposer needs to know what the data is about.")
            if not self.discovery.few_shot_path:
                raise ValueError("discovery.few_shot_path is required when discovery is "
                                 "enabled: the example rows are precomputed by the "
                                 "dataset's prepare step.")
        indicators = self.data.missing_indicators
        if indicators.scope not in ("candidates", "all"):
            raise ValueError("data.missing_indicators.scope must be candidates|all, "
                             f"got {indicators.scope!r}")
        if indicators.treat_as not in ("new", "source"):
            raise ValueError("data.missing_indicators.treat_as must be new|source, "
                             f"got {indicators.treat_as!r}")
        if not 0.0 <= indicators.min_missing_rate < 1.0:
            raise ValueError("data.missing_indicators.min_missing_rate must be in [0, 1), "
                             f"got {indicators.min_missing_rate}")
        if not indicators.suffix:
            raise ValueError("data.missing_indicators.suffix must not be empty")
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
