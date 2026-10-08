import { useEffect, useMemo, useRef, useState } from 'react'
import { api, streamUrl, useEventStream, usePolled } from '../api'
import { deriveEval } from '../derive'
import type { EvalView, PoolFeature } from '../types'
import { ConfirmButton } from './ConfirmButton'
import s from './EvalPanel.module.css'

export const signed = (x: number | null | undefined, digits = 4) =>
  x == null ? '–' : `${x >= 0 ? '+' : ''}${x.toFixed(digits)}`
export const pct = (x: number | null | undefined) => (x == null ? '–' : `${(x * 100).toFixed(1)}%`)
const clock = (seconds: number) =>
  `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`
const tone = (x: number | null | undefined) => (x == null || x === 0 ? '' : x > 0 ? s.up : s.down)

// A verdict row's gate columns and their outcomes.
const GATE_VALUES = new Set(['PASS', 'FAIL', 'not evaluable'])

/**
 * One evaluation, or one variant of it, at the right of the results: the variant's
 * numbers, its verdict gate by gate, the feature itself, and how the evaluation ran.
 */
export function EvalPanel({ evalId, variant, onSelect, onClose, onDeleted }: {
  evalId: string
  variant: string | null
  onSelect: (variant: string | null) => void
  onClose: () => void
  onDeleted: () => void
}) {
  const { events, gone } = useEventStream(streamUrl.evaluation(evalId))
  const view = useMemo(() => deriveEval(events), [events])
  const [pool] = usePolled(api.features, [], 30000)
  const [now, setNow] = useState(Date.now() / 1000)
  useEffect(() => {
    if (view.status !== 'running') return
    const timer = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(timer)
  }, [view.status])
  const elapsed = (view.status === 'running' ? now : view.lastTs) - view.startTs

  const row = variant ? view.rows.find((r) => r.variant === variant) : undefined
  const variants = view.rows.filter((r) => r.variant !== 'base')
  const when = view.startTs ? new Date(view.startTs * 1000).toLocaleString([], {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }) : ''

  if (gone) {
    return (
      <aside className={s.panel}>
        <div className={s.head}><span className={s.title}>Evaluation not found</span>
          <button className={s.close} onClick={onClose} title="Close">×</button></div>
        <div className={s.body}><div className={s.hint}>It was deleted.</div></div>
      </aside>
    )
  }

  return (
    <aside className={s.panel}>
      <div className={s.head}>
        <div className={s.headText}>
          <span className={s.kicker}>
            {row ? (row.variant.startsWith('combo__') ? 'Combination' : 'Feature') : 'Evaluation'}
            {' · '}{when}
          </span>
          <span className={s.title}>
            {row ? (row.variant.startsWith('combo__') ? `⊕ ${row.variant.slice(7)}` : row.variant.replace(/^loi__/, ''))
              : `${view.features.length} feature${view.features.length === 1 ? '' : 's'}`
                + (Object.keys(view.combinations).length ? ` · ${Object.keys(view.combinations).length} combination(s)` : '')}
          </span>
        </div>
        <span className={`${s.state} ${s[view.status]}`}>
          {view.status}{view.startTs > 0 && ` · ${clock(Math.max(0, elapsed))}`}
        </span>
        {view.status === 'running' && (
          <ConfirmButton label="Stop" confirm="Stop this evaluation?"
                         onConfirm={async () => { await api.stopEvaluation(evalId) }} />
        )}
        <button className={s.close} onClick={onClose} title="Close">×</button>
      </div>

      <div className={s.body}>
        {view.error && <div className={s.error}>{view.error}</div>}
        {row && view.status === 'done' && <VariantDetail view={view} row={row} pool={pool ?? []} />}
        {variant && !row && view.status === 'done' && (
          <div className={s.hint}>This evaluation has no variant {variant}.</div>
        )}

        {variants.length > 0 && (
          <section className={s.section}>
            <h3 className={s.h}>{row ? 'Evaluated together' : 'Variants'}</h3>
            <div className={s.variants}>
              {variants.map((r) => (
                <button key={r.variant} onClick={() => onSelect(r.variant)}
                        className={`${s.variantRow} ${r.variant === variant ? s.variantOn : ''}`}>
                  <span className={s.mono}>{r.variant.startsWith('combo__') ? `⊕ ${r.variant.slice(7)}` : r.variant.replace(/^loi__/, '')}</span>
                  <span className={`${s.num} ${tone(r.gini_gain_test)}`}>{signed(r.gini_gain_test)}</span>
                </button>
              ))}
            </div>
          </section>
        )}

        <section className={s.section}>
          <Progress view={view} />
        </section>

        <div className={s.foot}>
          {view.outputDir && <span className={s.hint}>Full report: <span className={s.mono}>{view.outputDir}</span></span>}
          {view.status !== 'running' && view.status !== 'idle' && (
            <ConfirmButton label="Delete evaluation" confirm="Delete this evaluation?"
                           onConfirm={async () => { await api.deleteEvaluation(evalId); onDeleted() }} />
          )}
        </div>
      </div>
    </aside>
  )
}

/** A variant's verdict gate by gate, and the feature (or members) behind it - its
 *  numbers are in the results table beside it. */
function VariantDetail({ view, row, pool }: {
  view: EvalView; row: EvalView['rows'][number]; pool: PoolFeature[]
}) {
  const combo = row.variant.startsWith('combo__')
  const column = row.variant.replace(/^(loi|combo)__/, '')
  const feature = combo ? undefined : view.features.find((f) => f.column === column)
  const members = combo ? (view.combinations[column]?.columns ?? []) : []
  const verdict = combo ? undefined : view.verdicts.find((v) => v.feature === column)
  const ranks = view.shapRanks[row.variant] ?? []
  const code = feature ? pool.find((p) => p.key === feature.key) : undefined

  return (
    <>
      {verdict && (
        <section className={s.section}>
          <h3 className={s.h}>Verdict <span className={verdict.verdict === 'PASS' ? s.pass : s.fail}>{verdict.verdict}</span></h3>
          {verdict.reason && <div className={s.text}>{verdict.reason}</div>}
          <div className={s.gates}>
            {Object.entries(verdict).filter(([k, v]) => k !== 'verdict' && GATE_VALUES.has(String(v))).map(([k, v]) => (
              <div key={k} className={s.gate}>
                <span>{k}</span>
                <span className={v === 'PASS' ? s.pass : v === 'FAIL' ? s.failStrong : s.dim}>
                  {v === 'not evaluable' ? 'not applied' : String(v)}</span>
              </div>
            ))}
          </div>
          <div className={s.hint}>
            “Not applied”: the gate had no evidence - its stage is off in the configuration (data
            quality), or the evaluation ran before the gate could read each feature's own model.
          </div>
        </section>
      )}

      {combo && (
        <section className={s.section}>
          <h3 className={s.h}>Members · SHAP rank in the combined model</h3>
          <div className={s.gates}>
            {members.map((m) => {
              const rank = ranks.find((k) => k.feature === m)
              const st = view.featureStats[m]
              return (
                <div key={m} className={s.gate}>
                  <span className={s.mono}>{m}</span>
                  <span>
                    {st && <span className={s.dim}>missing {pct(st.missing_rate)} · </span>}
                    {rank ? <><b>#{rank.rank}</b><span className={s.dim}> of {rank.of}</span></> : <span className={s.dim}>–</span>}
                  </span>
                </div>
              )
            })}
          </div>
        </section>
      )}

      {feature && (
        <section className={s.section}>
          <h3 className={s.h}>Feature</h3>
          <div className={s.text}>{feature.description}</div>
          <div className={s.hint}>{feature.level} · from “{feature.direction}”{code?.sources.length ? ` · ${code.sources.join(' + ')}` : ''}</div>
          {code
            ? <details className={s.code}><summary>feature script</summary><pre>{code.code}</pre></details>
            : <div className={s.hint}>The feature is no longer in the pool; its script is in the evaluation's report.</div>}
        </section>
      )}
    </>
  )
}

// Each step of an evaluation in plain words: what it does, and on which rows.
const STEP_TEXT: Record<string, { title: string; what: string }> = {
  features: { title: 'Compute the features',
    what: "Each feature's script runs on every train, valid and test row, through the linkage it was verified with." },
  data: { title: 'Check the data',
    what: 'Splits, target, ids and feature types are checked before any model trains.' },
  feature_selection: { title: 'Screen the features',
    what: 'Spearman and mutual-information screens on train. A flagged feature still goes on unless the gates are enforced.' },
  modeling: { title: 'Train the models',
    what: 'XGBoost on train: base, base + each feature, and base + each combination. Valid stops the training.' },
  analysis: { title: 'Score on test',
    what: 'Gini, capture rates and SHAP ranks on test - the out-of-time hold-out, used here for the first time.' },
  verdict: { title: 'Apply the gates',
    what: "Each feature's test result against the evaluation gates: PASS or FAIL." },
}

// What older evaluations reported as warnings, though every evaluation has them by design.
const BY_DESIGN = ['No model with every chosen feature together', 'The SHAP gate needs that all-together model',
                   'Some gates had no evidence']

/** How the evaluation went, step by step: what each step does, its result, and why it warned. */
function Progress({ view }: { view: EvalView }) {
  const [showLog, setShowLog] = useState(false)
  const logRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [view.logs.length, showLog])

  const scriptsDone = view.codes.filter((c) => c.state !== 'running').length
  const scriptState = !view.codes.length ? (view.status === 'running' ? 'running' : 'pending')
    : view.codes.some((c) => c.state === 'error') ? 'failed'
    : scriptsDone < view.codes.length || !view.stages.length ? 'running' : 'passed'
  // The gates a verdict could not apply - no evidence for them in an evaluation.
  const notApplied = [...new Set(view.verdicts.flatMap((v) =>
    Object.entries(v).filter(([, x]) => x === 'not evaluable').map(([k]) => k)))]
  const steps = [
    { key: 'features', label: 'Features', status: scriptState, error: null,
      detail: `${scriptsDone}/${view.codes.length} scripts`, elapsed_seconds: null, warnings: [] as string[] },
    ...view.stages.filter((st) => st.status !== 'skipped').map((st) => {
      const warnings = (st.warnings ?? []).filter((w) => !BY_DESIGN.some((b) => w.startsWith(b)))
      const designOnly = st.status === 'warning' && st.warnings?.length && !warnings.length
      return { ...st, warnings, status: designOnly ? 'passed' as const : st.status }
    }),
  ]

  return (
    <div className={s.progress}>
      <h3 className={s.h}>How it was evaluated</h3>
      <ol className={s.steps}>
        {steps.map((st, i) => {
          const text = STEP_TEXT[st.key] ?? { title: st.label, what: '' }
          const warnings = st.warnings.length ? st.warnings
            : st.status === 'warning' ? ['Finished with a warning - see the pipeline log.'] : []
          return (
            <li key={st.key} className={`${s.stepRow} ${s['row_' + st.status]}`}>
              <span className={s.stepNum}>{i + 1}</span>
              <div className={s.stepMain}>
                <div className={s.stepTop}>
                  <span className={s.stepTitle}>{text.title}</span>
                  <span className={s.stepResult}>
                    {st.status === 'running' ? 'running…' : st.status === 'pending' ? 'waiting' : st.detail}
                    {st.elapsed_seconds != null && <span className={s.dim}> · {st.elapsed_seconds.toFixed(1)}s</span>}
                  </span>
                </div>
                {text.what && <div className={s.stepWhat}>{text.what}</div>}
                {st.key === 'features' && view.codes.length > 0 && (
                  <div className={s.scripts}>
                    {view.codes.map((c) => (
                      <div key={c.code_id} className={s.script}>
                        <span className={`${s.dot} ${s[c.state]}`} />{c.title}
                        <span className={s.scriptTime}>{c.state === 'running' ? '…' : `${c.elapsed_s ?? 0}s`}</span>
                      </div>
                    ))}
                  </div>
                )}
                {st.error && <div className={s.stepWarn}>{st.error}</div>}
                {warnings.map((w, j) => <div key={j} className={s.stepWarn}>{w}</div>)}
                {st.key === 'verdict' && notApplied.length > 0 && (
                  <div className={s.stepNote}>Not applied, for lack of evidence: {notApplied.join(', ')}.</div>
                )}
              </div>
            </li>
          )
        })}
      </ol>
      <button className={s.logToggle} onClick={() => setShowLog(!showLog)}>
        {showLog ? '▾' : '▸'} Pipeline log ({view.logs.length} lines)
      </button>
      {showLog && (
        <div className={s.log} ref={logRef}>
          {view.logs.length === 0 && <span className={s.dim}>waiting for the pipeline…</span>}
          {view.logs.map((l) => (
            <div key={l.seq} className={`${s.logLine} ${s['log_' + l.level] ?? ''}`}>
              <span className={s.logTime}>{clock(Math.max(0, l.ts - view.startTs))}</span>{l.message}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
