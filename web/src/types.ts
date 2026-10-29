// The event streams from agent/session.py and agent/evaluate.py, and the
// views derived from them.

export type Ev = {
  seq: number
  event: string
  run_id: string
  ts: number
  [key: string]: any
}

export type SourceState = 'linked' | 'needs_linkage' | 'schema_only'

export type Source = {
  name: string
  usable: boolean
  state: SourceState
  registered: boolean
  data: string | null
  columns: Record<string, string>
}

export type SetupValue = string | string[] | number | number[] | boolean

export type SetupField = {
  key: string
  label: string
  kind: 'path' | 'file' | 'files' | 'text' | 'longtext' | 'number' | 'bool' | 'select' | 'numbers'
  upload?: 'context' | 'source_schema' | 'scope_file' | 'scope_note' | 'shot_spec' | 'shot_table'
  options?: string[]
  int?: boolean
  section?: string
  source_file?: string | null    // a target cases history: its dataset's own file name
  help: string
  value: SetupValue
  default: SetupValue
  changed: boolean
  block: string
}

export type SetupBlock = {
  key: 'task' | 'model_folder' | 'model' | 'shots' | 'context' | 'folder' | 'additional' | 'scope' | 'evaluation'
    | 'cases_folder' | 'target_cases'
  title: string
  help: string
  folder: boolean
  fields: SetupField[]
}

export type SetupView = {
  config_file: string
  blocks: SetupBlock[]
  upload_limit: number
  error: string | null
}

export type PathCheck = Record<string, { exists: boolean; dir: boolean; bytes: number | null }>

/** One category of example rows: the clustering shots, or one of the user's spec files. */
export type ShotCategory = {
  key: string
  kind: 'clustering' | 'ids' | 'table' | 'error'
  name: string
  context: string
  rotate: boolean
  batch_size: number | null
  batches: number
  per_batch: number[]
  path: string | null
  ids: number
  found: number
  columns: number
  classes: Record<string, number>
  missing: string[]
  n_missing: number
}

export type Level = 'L1' | 'L2' | 'L3'

export type Params = {
  K: number
  model: string
  engine: string
  min_gini_gain: number
  min_capture_gain: number | null   // null = not gated
  capture_percent?: number          // the capture rate's top share, from the config
  levels: Level[]
  sources: string[]
  ideas_per_round?: number          // ideas asked for at a time; more rounds for a large K
}

export type Workspace = {
  name: string
  trace_viewer?: string | null      // the trace viewer started beside the server, if any
  base_features: number
  id_column: string
  id_format: string
  target: string
  screen_rows: { fit: number; scored: number }
  additional_data_dir: string | null
  sources: Source[]
  scope: Record<string, number>
  scopes: Record<string, Record<string, number>>   // by keyword: status -> variables
  scope_files: { scope: string; name: string; path: string }[]
  scope_notes: { name: string; chars: number }[]
  shots: ShotCategory[]
  defaults: Omit<Params, 'sources'> & { level_weights?: number[] }
  choices: { models: string[]; engines: string[]; levels: Level[] }
  active_run: string | null
  queued_runs?: number                // runs waiting in line for room
  active_linkage: string | null
}

/** A run asked for while there was no room: started, in order, as runs end. */
export type QueuedRun = {
  queue_id: string
  direction: string
  explore: boolean
  queued: number
  error: string | null                // it could not start: kept in line, skipped
}

export type RunSummary = {
  run_id: string
  direction: string
  K: number
  started: number
  params: Params | null
  quota?: Record<string, number>
  maxAttempts?: number                // K is a target of results; attempts are capped
  verified: number
  mode: 'features' | 'l3'
  requests: number
  status: 'running' | 'done' | 'unfinished'
  stopped_because: string | null
  explore?: boolean                   // an open exploration: no direction from the user
}

export type CodeCard = {
  kind: 'code'
  seq: number
  code_id: string
  intent: string
  mode: 'probe' | 'linkage' | 'aggregate' | 'feature'
  level: string | null
  title: string
  code: string
  state: 'running' | 'ok' | 'error' | 'stopped'
  stdout?: string
  error?: string | null
  result?: { n_rows: number; columns: Record<string, string>; head: string } | null
  elapsed_s?: number
}

export type ApprovalCard = {
  kind: 'approval'
  seq: number
  req_id: string
  type: 'linkage' | 'data_pull'
  payload: Ev
  resolved?: { approved: boolean; note: string }
}

export type LedgerRow = {
  intent: string
  name: string
  description: string
  level: string
  code_id: string
  sources: string[]
  delta: number | null              // the Gini gain
  capture_gain: number | null
  base_score: number | null
  coverage: number | null
  verified: boolean
  reason: string
  deleted?: boolean
}

export type TraceItem =
  | { kind: 'message'; seq: number; text: string }
  | { kind: 'tool'; seq: number; tool: string; args: string }
  | { kind: 'skill'; seq: number; name: string }
  | { kind: 'note'; seq: number; tone: 'info' | 'error' | 'done'; text: string }
  | { kind: 'ideas'; seq: number; focus: string[]; ideas: Idea[]
      used?: Record<string, string>     // idea name -> the intent that proposed it (I3, R1)
      sentBack?: number                 // a try the checks returned: its attempt number
      missed?: string[] }               // recorded anyway, last resort: the rules it missed
  | CodeCard
  | ApprovalCard

/** A data-request run's output: why the data is needed, and the SQL to pull it. */
export type DataRequest = {
  intent: string
  source_name: string
  gap: string
  features: string
  sql: string                  // empty beyond scope
  tables: string[]
  columns?: string[]
  scope?: string               // the scope it reads, by keyword (CAS, say), or 'beyond_scope' (an idea)
  data?: string                // beyond scope: the data it needs and where it would come from
  refunded?: boolean           // dropped as covered by current data - its intent came back
  deleted?: boolean            // taken off the request set by the user
  status: 'proposed' | 'kept' | 'dropped' | string
  challenge?: {
    verdict: 'constructible' | 'partly' | 'new' | 'unchallenged'
    reasoning: string
    columns: string[]
    code: string
    code_ok: boolean | null
    code_error?: string | null
    note?: string
  }
}

/** One idea of the brainstorm. Older runs carry a hypothesis and no name. */
export type Idea = { n: number; name?: string; level?: string; lens: string
                     engine?: string                                   // the idea engine that made it
                     measures?: string[]                               // ways to measure it
                     description?: string; hypothesis?: string; data: string
                     beyond_scope?: boolean; beyond_cas?: boolean }   // beyond_cas: older runs

export type StepKind = 'explore' | 'ideas' | 'linkage' | 'aggregate' | 'intent' | 'l3' | 'challenge' | 'stage' | 'summary'

/** One tool call: which tool, whether it went through, and the model response it came in -
 *  calls that share a `batch` were released together, in parallel. */
export type ToolUse = { call_id: string; tool: string; batch: number; parallel: number; ok: boolean }

/** A data request's try that validation sent back - before the one that was recorded. */
export type RefusedTry = { sql: string; error: string }
export type StepStatus = 'running' | 'waiting' | 'done' | 'failed' | 'verified' | 'rejected'
  | 'sent_back'   // a request the checks returned to fix - nothing was spent

/** One stage of the run: what the trace groups and the timeline lists. */
export type Step = {
  id: number
  kind: StepKind
  title: string
  status: StepStatus
  items: TraceItem[]
  startTs: number
  endTs: number
  level?: string
  row?: LedgerRow
  request?: DataRequest
  challengeOf?: string
  /** an aggregate step: the aggregate's name - its scripts arrive before its call */
  aggregateOf?: string
  note?: string
  summary?: string
  tools?: ToolUse[]
  refused?: RefusedTry[]
}

export type RunView = {
  direction: string
  K: number
  params: Params | null
  quota?: Record<string, number>
  maxAttempts?: number                // K is a target of results; attempts are capped
  startTs: number
  steps: Step[]
  ledger: LedgerRow[]
  requests: DataRequest[]
  skills: string[]
  pending: ApprovalCard[]
  status: 'idle' | 'running' | 'waiting' | 'done' | 'error'
  stoppedBecause: string | null
  codeCount: number
  explore: boolean                    // no direction: the agent draws a theme per round
  themes: { round: number; theme: string; exploredBefore: number }[]
}

export type PoolFeature = {
  key: string
  run_id: string
  direction: string
  name: string
  description: string
  level: string
  sources: string[]
  delta: number
  coverage: number | null
  code: string
  missing_linkage: string[]
  linkage: Record<string, string>
  // What evaluations made of it: the latest alone, and the combinations it was in.
  evaluation?: {
    single: { eval_id: string; evaluated: number; verdict: string | null; reason: string
              gini_gain: number | null; trained: boolean; not_trained: string | null } | null
    combinations: string[]
  } | null
}

// One row of the logging board (GET /api/board): a feature or a data request
// with the idea it tests - or an idea that made nothing yet. Written by the
// server from the run's records, not from the agent's account of them.
export type BoardRow = {
  feature_id: string        // <run_id>:<name> for a feature - the key Metrics results carries; '' otherwise
  run_id: string
  direction: string
  idea: string
  round: number | null
  engine: string
  lens: string
  idea_status: '' | 'explore' | 'active' | 'low' | 'retired'
  p_useful: number | null
  idea_description: string
  idea_measures: string[]   // the ways the idea can be measured, as the agent gave them
  idea_same_as?: string     // <run_id>:<idea> it is linked to - its standing pools that family's evidence
  kind: 'feature' | 'request' | 'idea'
  name: string
  level: string | null
  description: string
  differs: string
  code: string
  attempt: string | null
  seen_before: string
  max_model_rho: number | null   // C1: the largest |Spearman rho| with a model column
  nearest_rho: number | null     // ... its sign kept: the rho with `nearest`
  nearest: string
  residual_rho: number | null    // C1's exception: rank correlation with the base model's residual
  variable_gini?: number | null  // C2: the feature alone as a score for y
  note: string
  gini_gain: number | null
  capture_gain: number | null
  status: string
  reason: string
  evidence: 'signal' | 'none' | null
  // A verified feature only: '' for a failed feature, a request or an idea alone.
  sent: '' | 'not evaluated' | 'evaluated' | 'evaluated · removed' | 'removed'
}

// One model call of a run (GET /api/runs/<id>/llm_calls): where in the run, how
// long, how many tokens - and, in full (GET /api/llm_calls/<n>), what it saw and said.
export type LlmCall = {
  id: number
  run_id: string
  stage: 'ideas' | 'propose' | 'linkage' | 'themes' | string
  round: number | null
  call: number
  agent: string
  model: string
  started_at: number
  ended_at: number | null
  duration_ms: number | null
  input_items: number | null
  input_chars: number | null
  input_tokens: number | null
  cached_tokens: number | null
  output_tokens: number | null
  reasoning_tokens: number | null
  tokens_estimated: number
  tool_calls: string[]
  outcome: 'ok' | 'unfinished' | string
  output_preview: string
}
export type LlmCallFull = LlmCall & {
  system_prompt: string
  input_json: unknown[] | { _elided: string; size: number } | null
  output_json: unknown[] | { _elided: string; size: number } | null
}

export type EvalSummary = {
  eval_id: string
  started: number
  status: 'running' | 'done' | 'error' | 'stopped'
  features: { key: string; column: string; name: string; direction: string
              level?: string; delta?: number | null; description?: string }[]
  combinations: Record<string, { keys: string[]; columns: string[]; auto?: boolean }>
  together?: 'all' | 'passing' | null      // then base + those singles in one model
  together_note?: string | null            // why that model was not trained
}

export type EvalStage = {
  key: string
  label: string
  status: 'pending' | 'running' | 'passed' | 'warning' | 'skipped' | 'failed' | 'stopped' | 'not_run'
  detail: string
  elapsed_seconds: number | null
  error: string | null
  warnings?: string[]          // the checks that fell short, in words
}

export type ShapRank = { feature: string; rank: number; of: number; share: number }

export type EvalView = {
  status: 'idle' | 'running' | 'done' | 'error' | 'stopped'
  startTs: number
  lastTs: number
  features: EvalSummary['features']
  combinations: EvalSummary['combinations']
  codes: CodeCard[]
  stages: EvalStage[]
  logs: { seq: number; ts: number; message: string; level: string }[]
  rows: (Record<string, any> & { variant: string; note?: string })[]
  verdicts: Record<string, any>[]
  capturePercents: string[]
  shapRanks: Record<string, ShapRank[]>
  featureStats: Record<string, FeatureStats>
  error?: string
  outputDir?: string
  gates?: string                     // open | enforce, as the evaluation ran
  modelErrors: Record<string, string>
  removed: string[]                  // variants taken off the results
}

/** The base model, as the latest finished evaluation trained it. */
export type BaseModelCard = {
  eval_id: string
  evaluated: number
  target: string
  n_features: number
  best_iteration: number | null
  splits: string[]
  gini: Record<string, number | null>
  capture: Record<string, Record<string, number | null>>
  data: Record<string, { rows: number; default_rate: number }>
  drivers: { feature: string; share: number }[]
  params: { max_depth: number | null; eta: number | null; num_boost_round: number | null }
}

/** One evaluated variant - base + a feature, or base + a combination - from one evaluation. */
export type EvalResult = {
  eval_id: string
  evaluated: number
  variant: string
  kind: 'feature' | 'combination'
  name: string
  members: string[]
  member_keys?: string[]
  auto?: boolean                      // the combination the evaluation added: every passing feature
  key: string | null
  run_id: string | null
  direction: string
  level: string | null
  description: string
  screen_gain: number | null
  gini_gain: number | null
  capture_gain: Record<string, number | null>
  shap: ShapRank[]
  verdict: string | null
  reason: string
  missing_rate?: number | null    // share of rows with no value, every split
  max_corr?: number | null        // largest |Spearman| with a base feature, on train
  max_corr_with?: string | null   // that base feature
  variable_gini?: number | null   // the feature alone as a score for the target, |2·AUC - 1| on test
  variable_gini_note?: string | null   // why there is none on test: empty, or one value there
  screen_variable_gini?: number | null // the same on the screen rows - check C2 in discovery
  trained?: boolean               // false: no model - its reason in not_trained
  not_trained?: string | null
  score_base?: number | null      // mean predicted probability on test: base's
  score_mean?: number | null      // ... and this model's
  score_change?: number | null    // their relative change
  nearest_shap?: { feature: string; base: number | null; variant: number | null } | null
}

// ------------------------------------------------------------------ inference

/** The target cases as set up, and what reading them found. */
export type TargetCases = {
  rows_path: string | null
  id_template: string | null
  case_column: string | null
  time_column: string | null
  flag_top: number
  // Each source of the additional data: the cases' own history of it, and how many
  // verified features read it.
  sources: Record<string, { path: string | null; state: 'provided' | 'not_found' | 'not_set'; needed_by: number }>
  error: string | null
  rows?: number
  cases?: number
  columns?: number
  id_example?: string | null
  model_id_example?: string
  missing_base?: string[]
  per_case?: { min: number; max: number } | null
}

export type CaseOutcome = 'earlier' | 'same' | 'later' | 'only_variant' | 'only_base' | 'neither'

export type ScoredVariant = { variant: string; kind: 'feature' | 'combination'; name: string;
                              columns: string[]; gini_gain: number | null }

/** A finished evaluation and its inference on the target cases - none yet, or how it went. */
export type InferenceSummary = {
  eval_id: string
  evaluated: number
  variants: ScoredVariant[]
  status: 'none' | 'running' | 'done' | 'error' | 'stopped'
  scored: number | null
  error: string | null
  sooner: Record<string, number>       // per variant: cases it flags sooner than base
}

export type CaseResult = {
  case: string
  rows: number
  first_base: number | null
  first_variant: number | null
  outcome: CaseOutcome
  rows_sooner: number | null
  days_sooner: number | null
  defaulted: number | null
}

export type TimelinePoint = {
  t: number; id: string; time: string
  y?: number | null                    // the outcome, when the cases carry it
  scores: Record<string, number>       // base and each variant
  values: Record<string, number | null>
}

export type InferenceDetail = {
  infer_id: string
  eval_id: string
  variants: ScoredVariant[]
  columns: string[]
  flag_top: number
  ts: number
  status: 'running' | 'done' | 'error' | 'stopped'
  error: string | null
  steps: { title?: string; state?: string; elapsed_s?: number }[]
  logs: string[]
  done: null | {
    cutoffs: Record<string, number>
    quantiles?: Record<string, Record<string, number>>   // model -> share -> its test cutoff
    sources: Record<string, string>
    rows_path: string
    summaries: Record<string, { cases: number; outcomes: Record<CaseOutcome, number>; per_case: CaseResult[] }>
  }
  timelines?: Record<string, TimelinePoint[]>
}

export type FeatureStats = { missing_rate: number | null; max_corr: number | null; max_corr_with: string | null; variable_gini?: number | null }

/** A kept data request in the request set - from any run. */
export type RequestItem = DataRequest & {
  key: string                  // run_id:intent
  run_id: string
  direction: string
  kind: 'in_scope' | 'beyond_scope'
  scope: string                // its scope's keyword, or 'beyond_scope'
}

export const BEYOND = 'beyond_scope'

/** A request's scope as runs record it now: older runs said cas / beyond_cas. */
export function scopeOf(raw: Record<string, any>): string {
  const v = raw.scope
  if (v === BEYOND || v === 'beyond_cas' || (!v && !raw.sql)) return BEYOND
  return !v || v === 'cas' ? 'CAS' : v
}

/** The theme pool an open exploration draws from - made from the task description. */
export type ThemePool = {
  themes: { theme: string; runs: number }[]
  made: number | null
  stale: boolean                      // made for another task description
  ready: boolean                      // the task is described
  generating: boolean
  error: string | null
}
