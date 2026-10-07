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
  help: string
  value: SetupValue
  default: SetupValue
  changed: boolean
  block: string
}

export type SetupBlock = {
  key: 'model' | 'context' | 'shots' | 'evaluation' | 'additional' | 'scope'
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
  base_features: number
  id_column: string
  id_format: string
  target: string
  screen_rows: { fit: number; scored: number }
  additional_data_dir: string | null
  sources: Source[]
  scope: Record<string, number>
  scope_files: { name: string; path: string }[]
  scope_notes: { name: string; chars: number }[]
  shots: ShotCategory[]
  defaults: Omit<Params, 'sources'> & { level_weights?: number[] }
  choices: { models: string[]; engines: string[]; levels: Level[] }
  active_run: string | null
  active_linkage: string | null
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
}

export type CodeCard = {
  kind: 'code'
  seq: number
  code_id: string
  intent: string
  mode: 'probe' | 'linkage' | 'feature'
  level: string | null
  title: string
  code: string
  state: 'running' | 'ok' | 'error'
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
      used?: Record<string, string> }   // idea name -> the intent that proposed it (I3, R1)
  | CodeCard
  | ApprovalCard

/** A data-request run's output: why the data is needed, and the SQL to pull it. */
export type DataRequest = {
  intent: string
  source_name: string
  gap: string
  features: string
  sql: string                  // empty beyond the CAS scope
  tables: string[]
  columns?: string[]
  scope?: 'cas' | 'beyond_cas'  // within the CAS scope (SQL, screened) or beyond it (an idea)
  data?: string                // beyond CAS: the data it needs and where it would come from
  refunded?: boolean           // dropped as covered by current data - its intent came back
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
                     description?: string; hypothesis?: string; data: string; beyond_cas?: boolean }

export type StepKind = 'explore' | 'ideas' | 'linkage' | 'intent' | 'l3' | 'challenge' | 'stage' | 'summary'

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
}

export type EvalSummary = {
  eval_id: string
  started: number
  status: 'running' | 'done' | 'error'
  features: { key: string; column: string; name: string; direction: string
              level?: string; delta?: number | null; description?: string }[]
  combinations: Record<string, { keys: string[]; columns: string[] }>
}

export type EvalStage = {
  key: string
  label: string
  status: 'pending' | 'running' | 'passed' | 'warning' | 'skipped' | 'failed'
  detail: string
  elapsed_seconds: number | null
  error: string | null
  warnings?: string[]          // the checks that fell short, in words
}

export type ShapRank = { feature: string; rank: number; of: number; share: number }

export type EvalView = {
  status: 'idle' | 'running' | 'done' | 'error'
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
}

/** One evaluated variant - base + a feature, or base + a combination - from one evaluation. */
export type EvalResult = {
  eval_id: string
  evaluated: number
  variant: string
  kind: 'feature' | 'combination'
  name: string
  members: string[]
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
}

export type FeatureStats = { missing_rate: number | null; max_corr: number | null; max_corr_with: string | null }
