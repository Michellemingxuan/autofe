import type { CodeCard, DataRequest, RunView, Step } from '../types'
import { ToolChips } from './ToolChips'
import s from './Timeline.module.css'

const KIND_LABEL: Record<Step['kind'], string> = {
  explore: 'look at the data', ideas: 'ideas', linkage: 'linkage', intent: 'intent', l3: 'data request',
  challenge: 'challenge', stage: 'stage', summary: 'summary',
}

const STATUS_LABEL: Record<Step['status'], string> = {
  running: 'working', waiting: 'waiting for you', done: 'done', failed: 'failed',
  verified: 'verified', rejected: 'not verified', sent_back: 'sent back · no cost',
}

// In a data-request run a step is a request: kept, dropped, proposed, or sent back
// to fix - a check's refusal costs nothing and is not a failure.
const L3_STATUS: Partial<Record<Step['status'], string>> = {
  verified: 'kept', rejected: 'dropped', running: 'proposed', sent_back: 'sent back · no cost',
}

// Above this many results the target is one bar, not a cell per result.
const CELLS_UP_TO = 24

// The proposing tools: a step's stage line already says what they did, retries included.
const PROPOSING = new Set(['screen_feature', 'screen_request', 'propose_new_data', 'challenge_request'])

const clock = (seconds: number) => {
  const m = Math.floor(seconds / 60)
  const sec = Math.floor(seconds % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

/**
 * The run as a process, top to bottom: one node per step, in the order the
 * trace shows them, with the time each started. The step the reader is
 * looking at in the trace is marked; clicking a node scrolls the trace to it.
 */
export function Timeline({ view, inView, onSelect }: {
  view: RunView
  inView: number | null
  onSelect: (id: number) => void
}) {
  const l3 = view.params?.levels?.length === 1 && view.params.levels[0] === 'L3'
  // Results are verified features and kept requests - a mixed run has both kinds.
  const used = view.ledger.length + view.requests.length
  const verifiedNames = view.ledger.filter((r) => r.verified && !r.deleted).map((r) => `${r.intent} ${r.name}`)
  const keptNames = view.requests.filter((r) => r.status === 'kept').map((r) => `${r.intent} ${r.source_name}`)
  const goodNames = [...verifiedNames, ...keptNames]
  const good = goodNames.length
  const mixed = !l3 && !!view.params?.levels.includes('L3')
  const live = view.status === 'running' || view.status === 'waiting'
  return (
    <aside className={s.panel}>
      <div className={s.head}>
        <span className={`eyebrow ${s.title}`}>Timeline</span>
        <span className={s.meta}>{view.steps.length} steps · {view.codeCount} scripts</span>
      </div>

      <div className={s.budget}>
        {/* K is a target: the bar fills with results - verified features, kept requests. */}
        {(view.K || 0) > CELLS_UP_TO ? (
          <div className={s.track} title={`${good} of ${view.K}`}>
            <span className={s.fillOk} style={{ width: `${Math.min(100, (100 * good) / view.K)}%` }} />
          </div>
        ) : (
          <div className={s.bar}>
            {Array.from({ length: view.K || 0 }, (_, i) => (
              <span key={i} className={`${s.cell} ${i < good ? s.cellOk : ''}`}
                    title={i < good ? goodNames[i] : `result ${i + 1} of ${view.K}`} />
            ))}
          </div>
        )}
        <span className={s.budgetText}>
          {good}/{view.K} {l3 ? 'kept' : mixed ? `results (${verifiedNames.length} L1/L2, ${keptNames.length} L3)` : 'verified'} · {used}
          {view.maxAttempts ? `/${view.maxAttempts}` : ''} attempts
        </span>
      </div>

      <ol className={s.list}>
        {view.steps.map((step, i) => {
          const last = i === view.steps.length - 1
          return (
            <li key={step.id}
                className={`${s.node} ${s[step.status]} ${inView === step.id ? s.inView : ''}`}>
              <div className={s.rail}>
                <span className={`${s.dot} ${s[step.status]}`} />
                {!last && <span className={s.line} />}
              </div>
              <button className={s.content} onClick={() => onSelect(step.id)}>
                <div className={s.top}>
                  <span className={s.kind}>
                    {l3 && step.kind === 'l3' ? (step.request ? `${step.request.intent} · data request` : 'data request')
                      : step.kind === 'intent' && step.row ? `${step.row.intent} · ${step.level}`
                      : step.kind === 'challenge' ? `${step.challengeOf ?? ''} · challenge`
                      : step.request ? `${step.request.intent} · data request` : KIND_LABEL[step.kind]}
                  </span>
                  <span className={s.time}>{clock(step.startTs - view.startTs)}</span>
                </div>
                <div className={s.name}>{step.title}</div>
                <ToolChips tools={(step.tools ?? []).filter((u) =>
                  !(PROPOSING.has(u.tool) && ['intent', 'l3', 'challenge'].includes(step.kind)))} />
                {(step.kind === 'l3' || (step.kind === 'challenge' && step.request)) &&
                  <Progress request={step.request} live={live} />}
                {step.kind === 'intent' && <FeatureProgress step={step} />}
                <div className={s.status}>
                  {l3 && step.kind === 'l3' ? (step.request?.challenge
                      ? `${step.request.challenge.verdict} → ${step.request.status}`
                      : step.request?.refunded ? 'dropped · not counted'
                      : !step.request ? (live ? 'screening…' : 'did not pass the screen')
                      : L3_STATUS[step.status] ?? STATUS_LABEL[step.status])
                    : step.row?.deleted ? 'removed from pool'
                    : step.kind === 'challenge' && step.request?.challenge
                      ? `${step.request.challenge.verdict} → ${step.request.status}`
                    : step.kind === 'l3' && step.request?.challenge
                      ? `${step.request.challenge.verdict} → ${step.request.status}`
                    : step.kind === 'l3' && step.request ? (step.request.scope === 'beyond_cas'
                      ? 'proposed · beyond CAS' : `proposed · reads ${step.request.tables.join(', ')}`)
                    : step.kind === 'stage' ? step.note : STATUS_LABEL[step.status]}
                  {step.row?.delta != null && (
                    <span className={s.delta}> · Δ {step.row.delta >= 0 ? '+' : ''}{step.row.delta.toFixed(4)}</span>
                  )}
                </div>
              </button>
            </li>
          )
        })}
        {view.steps.length === 0 && <li className={s.empty}>Waiting for the agent…</li>}
      </ol>
    </aside>
  )
}

/**
 * A request's way through the run, on one line: proposed; screened - its SQL
 * validated (within the CAS scope only) (a try sent back is retried, shown as screening until one passes or
 * the run ends); challenged; kept or dropped.
 */
function Progress({ request: r, live }: { request?: DataRequest; live: boolean }) {
  const screened: Pip = r ? 'on' : live ? 'run' : 'fail'
  const challenged: Pip = r?.challenge && r.status !== 'proposed' ? 'on' : r && live ? 'run' : 'pending'
  const end: Pip = challenged !== 'on' ? 'pending' : r!.status === 'dropped' ? 'drop' : 'keep'
  const stages: [string, Pip][] = [
    ['proposed', 'on'],
    // Beyond the CAS scope there is no SQL to screen: proposed, then challenged.
    ...(r?.scope === 'beyond_cas' ? [] : [[screened === 'run' ? 'screening' : 'screened', screened] as [string, Pip]]),
    ['challenged', challenged],
    [end === 'drop' ? 'dropped' : end === 'keep' ? 'kept' : 'kept / dropped', end],
  ]
  return (
    <div className={s.progress}>
      {stages.map(([label, state], i) => (
        <span key={i} style={{ display: 'contents' }}>
          {i > 0 && <span className={s.pipLine} />}
          <span className={`${s.pip} ${PIP[state]}`}>{label}</span>
        </span>
      ))}
    </div>
  )
}

type Pip = 'pending' | 'on' | 'run' | 'fail' | 'keep' | 'drop'

/**
 * A feature's way through screen_feature, on one line: proposed; screened - the
 * checks before anything runs (name, level share, sources, linkage), which send a
 * proposal back at no cost; executed - its script built the column; validated -
 * the guards, the fit on screen_train and the gates on screen_valid. The stage
 * that stopped it is red; a proposal sent back first and then fixed says so.
 */
function FeatureProgress({ step }: { step: Step }) {
  const row = step.row
  const code = step.items.find((i): i is CodeCard => i.kind === 'code' && i.mode === 'feature')
  const sentBack = (step.tools ?? []).filter((u) => u.tool === 'screen_feature' && !u.ok).length
  const buildFailed = !!row && /^(script failed|build\(\) returned)/.test(row.reason ?? '')
  const scored = row?.delta != null
  // Past the checks once it holds an intent: a script started, or a result came back.
  const screened: Pip = row || code ? 'on' : 'run'
  const executed: Pip = screened !== 'on' ? 'pending'
    : buildFailed || code?.state === 'error' ? 'fail'
    : row || code?.state === 'ok' ? 'on' : 'run'
  const validated: Pip = executed !== 'on' ? 'pending'
    : scored ? (row!.verified ? 'keep' : 'drop') : row ? 'fail' : 'run'
  const stages: [string, Pip, string][] = [
    ['proposed', 'on', 'the agent proposed the feature and its code'],
    ['screened', screened, 'checks before anything runs: name, level share, sources, linkage'
      + (sentBack ? ` - sent back ${sentBack}× first, then fixed` : '')],
    ['executed', executed, 'the feature script ran and built the column'],
    ['validated', validated, 'guards (finite, no spikes, not redundant), the fit on screen_train, '
      + 'the Gini and capture gains on screen_valid against the gates'],
  ]
  return (
    <div className={s.progress}>
      {stages.map(([label, state, title], i) => (
        <span key={label} style={{ display: 'contents' }}>
          {i > 0 && <span className={s.pipLine} />}
          <span className={`${s.pip} ${PIP[state]}`} title={title}>
            {label}{label === 'screened' && sentBack > 0 && <span className={s.pipNote}> ↩{sentBack}</span>}
          </span>
        </span>
      ))}
    </div>
  )
}

const PIP: Record<Pip, string> = {
  pending: '', on: s.pipOn, run: `${s.pipOn} ${s.pipRun}`, fail: s.pipFail, keep: s.pipKeep, drop: s.pipDrop,
}
