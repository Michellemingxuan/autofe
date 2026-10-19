import { useEffect, useMemo, useState } from 'react'
import { api, usePolled } from '../api'
import { navigate } from '../App'
import type { InferenceDetail, TargetCases, TimelinePoint } from '../types'
import { signed } from './EvalPanel'
import s from './InferencePage.module.css'

const SHARES = [0.01, 0.02, 0.05, 0.1, 0.2]
const pctOf = (x: number) => `${+(x * 100).toFixed(1)}%`
const when = (ts: number) => new Date(ts * 1000).toLocaleString([], {
  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false,
})
const day = (t: string) => t.slice(0, 10)
const variantName = (v: string) => (v.startsWith('combo__') ? `⊕ ${v.slice(7)}` : v.replace(/^loi__/, ''))

// Sooner: base + new flags the case at an earlier row - or base never does.
// Later: at a later row - or base + new never does.
type Outcome = 'sooner' | 'same' | 'later' | 'neither'
const OUTCOME: Record<Outcome, string> = {
  sooner: 'sooner', same: 'same row', later: 'later', neither: 'neither flags',
}
const ORDER: Outcome[] = ['sooner', 'same', 'later', 'neither']

/** How a row is flagged: each model at its own top share of test, both at base's,
 *  or both at a fixed score. */
type Rule = { mode: 'quantile' | 'base' | 'score'; share: number; score: number }

/** A model of an evaluation, with every case scored by it and by base. */
type Model = { evalId: string; evaluated: number; variant: string; columns: string[]
               gini: number | null; detail: InferenceDetail }

type CaseView = {
  case: string; points: TimelinePoint[]
  first: { base: TimelinePoint | null; model: TimelinePoint | null }
  outcome: Outcome
  // Rows earlier than base (negative: later). When one model never flags the case,
  // the rows from the other's flag to the case's end - at least that many.
  rowsSooner: number | null
  daysSooner: number | null
}

function cutoffsOf(model: Model, rule: Rule): { base: number; model: number } | null {
  const done = model.detail.done!
  const q = done.quantiles
  const key = String(rule.share)
  const at = (name: string) => q?.[name]?.[key]
    ?? (Math.abs(rule.share - model.detail.flag_top) < 1e-9 ? done.cutoffs[name] : undefined)
  if (rule.mode === 'score') return { base: rule.score, model: rule.score }
  const base = at('base')
  const mine = rule.mode === 'base' ? base : at(model.variant)
  return base == null || mine == null ? null : { base, model: mine }
}

function casesOf(model: Model, cut: { base: number; model: number }): CaseView[] {
  return Object.entries(model.detail.timelines ?? {}).map(([name, points]) => {
    const base = points.find((p) => p.scores.base >= cut.base) ?? null
    const mine = points.find((p) => p.scores[model.variant] >= cut.model) ?? null
    const outcome: Outcome = !base && !mine ? 'neither' : !base ? 'sooner' : !mine ? 'later'
      : mine.t < base.t ? 'sooner' : mine.t > base.t ? 'later' : 'same'
    const days = base && mine ? (Date.parse(base.time) - Date.parse(mine.time)) / 864e5 : NaN
    return { case: name, points, first: { base, model: mine }, outcome,
             rowsSooner: base && mine ? base.t - mine.t : mine ? points.length + 1 - mine.t
               : base ? -(points.length + 1 - base.t) : null,
             daysSooner: Number.isFinite(days) ? Math.round(days) : null }
  })
}

const worse = (o: Outcome) => o === 'later'
/** How far ahead base + new is on a case, in words; "+" where a model never flags it. */
const ahead = (c: CaseView) => {
  const n = Math.abs(c.rowsSooner ?? 0), plus = c.first.base && c.first.model ? '' : '+'
  return c.outcome === 'sooner' ? `${n}${plus} row${n === 1 ? '' : 's'} earlier`
    : c.outcome === 'later' ? `${n}${plus} row${n === 1 ? '' : 's'} later` : OUTCOME[c.outcome]
}

/**
 * Step 3, inference results. When an evaluation ends, the target cases - the
 * ones the analyst cares about - are scored by its base model and by every model
 * it trained. Pick how a row is flagged; for each model, which cases it flags
 * sooner than base, and each case's scores row by row.
 */
export function InferencePage() {
  const [cases] = usePolled(api.targetCases, [], 30000)
  const [list, refreshList] = usePolled(api.inferences, [], 3000)
  const [details, setDetails] = useState<Record<string, InferenceDetail>>({})
  const [error, setError] = useState<string | null>(null)
  const [rule, setRule] = useState<Rule>({ mode: 'quantile', share: 0.05, score: 0.1 })
  const [pick, setPick] = useState<string | null>(null)          // `${evalId}|${variant}`

  useEffect(() => { if (cases) setRule((r) => ({ ...r, share: cases.flag_top })) }, [cases?.flag_top])
  // Each scored evaluation's scores - again while one is being scored.
  const scored = (list ?? []).filter((e) => e.status !== 'none')
  const stamp = scored.map((e) => `${e.eval_id}:${e.status}:${e.scored}`).join(',')
  useEffect(() => {
    let alive = true
    for (const e of scored) {
      api.inference(e.eval_id).then((d) => { if (alive) setDetails((old) => ({ ...old, [e.eval_id]: d })) })
        .catch(() => {})
    }
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stamp])

  // Each model by its latest evaluation (the list is newest first): scoring a model
  // again replaces its row.
  const models: Model[] = useMemo(() => {
    const seen = new Set<string>()
    return scored.flatMap((e) => {
      const d = details[e.eval_id]
      if (!d?.done) return []
      return d.variants.filter((v) => d.done!.summaries[v.variant]).map((v) => ({
        evalId: e.eval_id, evaluated: e.evaluated, variant: v.variant, columns: v.columns,
        gini: v.gini_gain, detail: d }))
    }).filter((m) => {
      const key = `${m.variant}|${[...m.columns].sort().join(',')}`
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
  }, [scored, details])
  const unscored = (list ?? []).filter((e) => e.status === 'none')
  const running = (list ?? []).filter((e) => e.status === 'running')
  const failed = (list ?? []).filter((e) => e.status === 'error')
  const blocked = !cases || !!cases.error || !!cases.missing_base?.length
  const scoreAll = async () => {
    setError(null)
    try { for (const e of unscored) await api.infer(e.eval_id) } catch (err: any) { setError(err.message) }
    refreshList()
  }
  const current = models.find((m) => `${m.evalId}|${m.variant}` === pick) ?? models[0]

  return (
    <div className={s.page}>
      <Header cases={cases} models={models.length} running={running.length} unscored={unscored.length}
              onScore={blocked ? undefined : scoreAll} />

      <div className={s.body}>
        {error && <div className={s.error}>{error}</div>}
        {failed.map((e) => (
          <div key={e.eval_id} className={s.error}>Scoring the evaluation of {when(e.evaluated)} failed: {e.error}
            {!blocked && <button className={s.link} onClick={async () => { await api.infer(e.eval_id); refreshList() }}>Score again</button>}
          </div>
        ))}
        {!models.length && (
          <div className={s.empty}>
            {running.length ? 'Scoring the target cases…'
              : unscored.length ? 'Score the target cases with the evaluated models - the button at the top right.'
              : 'Evaluate features: as each evaluation ends, its models score the target cases here.'}
          </div>
        )}

        {models.length > 0 && (
          <>
            <RulePicker rule={rule} onChange={setRule} model={current} />
            <ModelTable models={models} rule={rule} current={current}
                        onPick={(m) => setPick(`${m.evalId}|${m.variant}`)} />
            {current && <CaseExplorer key={`${current.evalId}|${current.variant}`} model={current} rule={rule} />}
          </>
        )}
      </div>
    </div>
  )
}

/** The target cases at a glance, like a run's header: what they are, and how many. */
function Header({ cases, models, running, unscored, onScore }: {
  cases: TargetCases | null; models: number; running: number; unscored: number; onScore?: () => void
}) {
  return (
    <header className={s.header}>
      <div className={s.headText}>
        <span className={`eyebrow ${s.eyebrow}`}>Evaluate · Inference results</span>
        <div className={s.titleRow}>
          <h1 className={s.title}>Target cases</h1>
          <span className={`${s.pill} ${running ? s.pillRun : models ? s.pillDone : ''}`}>
            <span className={s.pillDot} />
            {running ? 'scoring…' : models ? `scored by base + ${models} model${models === 1 ? '' : 's'}` : 'not scored'}
          </span>
        </div>
        {cases?.error && (
          <div className={s.desc}><span className={s.warn}>{cases.error}</span>{' '}
            <button className={s.link} onClick={() => navigate('setup')}>Set them up</button></div>
        )}
        {cases && !cases.error && Object.entries(cases.sources).some(([, x]) => x.needed_by && x.state !== 'provided') && (
          <div className={s.desc}><span className={s.warn}>No history for {Object.entries(cases.sources)
            .filter(([, x]) => x.needed_by && x.state !== 'provided').map(([n]) => n).join(', ')}, which the features read.</span>{' '}
            <button className={s.link} onClick={() => navigate('setup')}>Set it in Setup</button></div>
        )}
      </div>
      <div className={s.headRight}>
        {cases && !cases.error && (
          <>
            <div className={s.counter}><span className={s.num}>{cases.cases?.toLocaleString()}</span>
              <span className={s.of}>cases</span></div>
            <div className={s.counter}><span className={s.num}>{cases.rows?.toLocaleString()}</span>
              <span className={s.of}>transactions{cases.per_case ? ` · ${cases.per_case.min}–${cases.per_case.max} a case` : ''}</span></div>
          </>
        )}
        {unscored > 0 && (
          <button className={s.action} disabled={!onScore} onClick={onScore}
                  title="Evaluations that ended before the target cases were set up">
            Score {unscored} evaluation{unscored === 1 ? '' : 's'}</button>
        )}
      </div>
    </header>
  )
}

/** How a row is flagged - the analyst's choice, applied to every model at once. */
function RulePicker({ rule, onChange, model }: { rule: Rule; onChange: (r: Rule) => void; model?: Model }) {
  const cut = model ? cutoffsOf(model, rule) : null
  const MODES: { mode: Rule['mode']; label: string; hint: string }[] = [
    { mode: 'quantile', label: 'Each model\'s top share', hint: 'Each model flags its own top share of test scores - the same share, so the same number of test rows' },
    { mode: 'base', label: 'Base\'s cutoff, for both', hint: 'Both models flag above the score base flags at its top share of test - the bar the current model sets' },
    { mode: 'score', label: 'A fixed score', hint: 'Both models flag above one predicted probability' },
  ]
  return (
    <section className={s.ruleBar}>
      <span className={s.ruleLabel}>Flag a row at</span>
      <div className={s.segment} role="radiogroup">
        {MODES.map((m) => (
          <button key={m.mode} role="radio" aria-checked={rule.mode === m.mode} title={m.hint}
                  className={rule.mode === m.mode ? s.segOn : ''} onClick={() => onChange({ ...rule, mode: m.mode })}>
            {m.label}</button>
        ))}
      </div>
      {rule.mode !== 'score' ? (
        <label className={s.ruleValue}>top
          <select value={rule.share} onChange={(e) => onChange({ ...rule, share: Number(e.target.value) })}>
            {[...new Set([...SHARES, rule.share])].sort((a, b) => a - b).map((x) =>
              <option key={x} value={x}>{pctOf(x)}</option>)}
          </select>of test</label>
      ) : (
        <label className={s.ruleValue}>score ≥
          <input type="number" min={0} max={1} step={0.01} value={rule.score}
                 onChange={(e) => onChange({ ...rule, score: Number(e.target.value) })} /></label>
      )}
      <span className={s.ruleNote}>
        {cut ? <>base flags ≥ <b>{cut.base.toFixed(4)}</b> · base + new ≥ <b>{cut.model.toFixed(4)}</b></>
          : 'this evaluation kept no cutoff at that share - score it again'}
      </span>
    </section>
  )
}

/** Every evaluated model against base: how its flags move against base's, case by case. */
function ModelTable({ models, rule, current, onPick }: {
  models: Model[]; rule: Rule; current?: Model; onPick: (m: Model) => void
}) {
  return (
    <section className={s.card}>
      <div className={s.cardHead}>
        <span className={`eyebrow ${s.label}`}>Each model against base</span>
        <span className={s.meta}>click a model for its cases</span>
      </div>
      <table className={s.table}>
        <thead>
          <tr>
            <th>Model</th><th className={s.num}>Test Δ Gini</th><th className={s.barHead}>Cases, by when base + new flags them against base</th>
            <th className={s.num} title="Cases flagged at some row: base → base + new">Cases flagged</th>
            <th className={s.num} title="Over the cases base + new flags sooner: the median rows earlier">Median rows earlier</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const cut = cutoffsOf(m, rule)
            const rows = cut ? casesOf(m, cut) : []
            const count = (o: Outcome) => rows.filter((c) => c.outcome === o).length
            const flagged = (who: 'base' | 'model') => rows.filter((c) => c.first[who]).length
            const early = rows.filter((c) => c.outcome === 'sooner').map((c) => c.rowsSooner ?? 0).sort((a, b) => a - b)
            const median = early.length ? (early[(early.length - 1) >> 1] + early[early.length >> 1]) / 2 : null
            const on = current && m.evalId === current.evalId && m.variant === current.variant
            return (
              <tr key={`${m.evalId}|${m.variant}`} className={on ? s.rowOn : s.row} onClick={() => onPick(m)}>
                <td>
                  <div className={s.mono}>base + {variantName(m.variant)}</div>
                  <div className={s.meta}>evaluated {when(m.evaluated)}</div>
                </td>
                <td className={s.num}>{signed(m.gini)}</td>
                <td>{cut ? <OutcomeBar counts={ORDER.map((o) => [o, count(o)] as [Outcome, number])} total={rows.length} />
                         : <span className={s.meta}>no cutoff at this share</span>}</td>
                <td className={s.num}>{flagged('base')} → <b className={flagged('model') > flagged('base') ? s.goodInk
                  : flagged('model') < flagged('base') ? s.warnInk : ''}>{flagged('model')}</b>
                  <span className={s.meta}> / {rows.length}</span></td>
                <td className={s.num}>{median != null ? <b className={s.goodInk}>{median}</b> : '–'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </section>
  )
}

function OutcomeBar({ counts, total }: { counts: [Outcome, number][]; total: number }) {
  return (
    <div className={s.barCell}>
      <div className={s.bar}>
        {counts.filter(([, n]) => n).map(([o, n]) => (
          <span key={o} className={s['b_' + o]} style={{ flex: n }} title={`${OUTCOME[o]}: ${n}`} />
        ))}
      </div>
      <div className={s.barLegend}>
        {counts.filter(([, n]) => n).map(([o, n]) => (
          <span key={o}><i className={s['b_' + o]} />{n} {OUTCOME[o]}</span>
        ))}
        <span className={s.meta}>of {total}</span>
      </div>
    </div>
  )
}

/** The cases on the left - each a small picture of both scores; the chosen one's
 *  curves on the right. */
function CaseExplorer({ model, rule }: { model: Model; rule: Rule }) {
  const cut = cutoffsOf(model, rule)
  const all = useMemo(() => (cut ? casesOf(model, cut) : []), [model, cut?.base, cut?.model])
  const [filter, setFilter] = useState<'all' | 'sooner' | 'worse'>('all')
  const [open, setOpen] = useState<string | null>(null)
  if (!cut) return null
  const shown = all.filter((c) => filter === 'all' || (filter === 'sooner' && c.outcome === 'sooner')
    || (filter === 'worse' && worse(c.outcome)))
    .sort((a, b) => ORDER.indexOf(a.outcome) - ORDER.indexOf(b.outcome)
      || (b.rowsSooner ?? 0) - (a.rowsSooner ?? 0) || a.case.localeCompare(b.case))
  const chosen = all.find((c) => c.case === open) ?? shown[0]
  const most = Math.max(1, ...all.map((c) => c.rowsSooner ?? 0))
  const FILTERS = [['all', 'All', all.length], ['sooner', 'Earlier', all.filter((c) => c.outcome === 'sooner').length],
                   ['worse', 'Later', all.filter((c) => worse(c.outcome)).length]] as const

  return (
    <section className={s.explorer}>
      <div className={s.caseList}>
        <div className={s.cardHead}>
          <span className={`eyebrow ${s.label}`}>Cases · base + {variantName(model.variant)}</span>
        </div>
        <div className={s.filters}>
          {FILTERS.map(([key, label, n]) => (
            <button key={key} className={filter === key ? s.filterOn : ''} onClick={() => setFilter(key)}>
              {label} <b>{n}</b></button>
          ))}
        </div>
        <div className={s.caseScroll}>
          {shown.map((c) => (
            <button key={c.case} className={`${s.caseItem} ${chosen?.case === c.case ? s.caseOn : ''}`}
                    onClick={() => setOpen(c.case)}>
              <span className={s.mono}>{c.case}</span>
              <span className={s.lead}>
                {c.outcome === 'sooner' && <i style={{ width: `${((c.rowsSooner ?? 0) / most) * 100}%` }} />}
              </span>
              <span className={`${s.outcome} ${s['o_' + c.outcome]}`}>{ahead(c)}</span>
            </button>
          ))}
          {!shown.length && <div className={s.meta}>No case here.</div>}
        </div>
      </div>
      <div className={s.chartPane}>
        {chosen ? <CaseDetail c={chosen} model={model} cut={cut} /> : <div className={s.empty}>No case.</div>}
      </div>
    </section>
  )
}

/** One case: its facts, and both scores row by row against each cutoff. */
function CaseDetail({ c, model, cut }: { c: CaseView; model: Model; cut: { base: number; model: number } }) {
  const [hover, setHover] = useState<TimelinePoint | null>(null)
  const { points } = c
  const v = model.variant
  const W = 720, H = 260, L = 46, R = 14, T = 14, B = 30
  const top = Math.max(cut.base, cut.model, ...points.flatMap((p) => [p.scores.base, p.scores[v]])) * 1.1
  const x = (t: number) => L + ((t - 1) / Math.max(1, points.length - 1)) * (W - L - R)
  const y = (val: number) => T + (1 - val / top) * (H - T - B)
  const line = (of: (p: TimelinePoint) => number) => points.map((p, i) => `${i ? 'L' : 'M'}${x(p.t)},${y(of(p))}`).join(' ')
  const area = (of: (p: TimelinePoint) => number) => `${line(of)} L${x(points[points.length - 1].t)},${y(0)} L${x(1)},${y(0)} Z`
  const shown = hover ?? c.first.model ?? c.first.base ?? points[points.length - 1]
  const slot = (W - L - R) / Math.max(2, points.length)
  // A mark at either end is anchored inward, so it is not cut off.
  const anchor = (t: number) => (t === points.length ? 'end' : t === 1 ? 'start' : 'middle')
  const fact = (label: string, p: TimelinePoint | null) => (
    <div className={s.fact}><span>{label}</span>
      <b>{p ? `row ${p.t}` : 'never'}</b><span className={s.meta}>{p ? day(p.time) : `in ${points.length} rows`}</span></div>
  )
  return (
    <div className={s.detail}>
      <div className={s.detailHead}>
        <div>
          <div className={s.kicker}>case</div>
          <div className={s.detailTitle}>{c.case}</div>
        </div>
        <div className={s.facts}>
          {fact('base first flags', c.first.base)}
          {fact('base + new first flags', c.first.model)}
          <div className={`${s.fact} ${c.outcome === 'sooner' ? s.factGood : worse(c.outcome) ? s.factBad : ''}`}>
            <span>base + new</span>
            <b>{ahead(c)}</b>
            <span className={s.meta}>{!c.first.base && c.first.model ? 'base never flags it'
              : c.first.base && !c.first.model ? 'base + new never flags it'
              : c.daysSooner != null && c.outcome !== 'same'
              ? `${Math.abs(c.daysSooner)} days ${c.daysSooner >= 0 ? 'earlier' : 'later'}` : ' '}</span>
          </div>
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className={s.svg} onMouseLeave={() => setHover(null)} role="img"
           aria-label="the case's scores by row: base and base + new">
        {[0, top / 2, top].map((val) => (
          <g key={val}>
            <line x1={L} x2={W - R} y1={y(val)} y2={y(val)} className={s.grid} />
            <text x={L - 6} y={y(val) + 3} className={s.axis} textAnchor="end">{val.toFixed(2)}</text>
          </g>
        ))}
        {c.first.base && c.first.model && c.first.model.t !== c.first.base.t && (
          <rect x={Math.min(x(c.first.base.t), x(c.first.model.t))} y={T} height={H - T - B}
                width={Math.abs(x(c.first.base.t) - x(c.first.model.t))}
                className={c.first.model.t < c.first.base.t ? s.sparkGood : s.sparkBad} />
        )}
        <path d={area((p) => p.scores[v])} className={s.areaVar} />
        <line x1={L} x2={W - R} y1={y(cut.base)} y2={y(cut.base)} className={s.cutBase} />
        <line x1={L} x2={W - R} y1={y(cut.model)} y2={y(cut.model)} className={s.cutVar} />
        <text x={L + 4} y={y(cut.base) - 4} className={s.cutLabel}>base cutoff</text>
        {Math.abs(y(cut.model) - y(cut.base)) > 12 && (
          <text x={L + 4} y={y(cut.model) - 4} className={`${s.cutLabel} ${s.varInk}`}>base + new cutoff</text>
        )}
        <path d={line((p) => p.scores.base)} className={s.lineBase} />
        <path d={line((p) => p.scores[v])} className={s.lineVar} />
        {points.map((p) => (
          <g key={p.t}>
            <circle cx={x(p.t)} cy={y(p.scores.base)} r={p.scores.base >= cut.base ? 4 : 2.2} className={s.dotBase} />
            <circle cx={x(p.t)} cy={y(p.scores[v])} r={p.scores[v] >= cut.model ? 4 : 2.2} className={s.dotVar} />
            <rect x={x(p.t) - slot / 2} y={T} height={H - T - B} width={slot} fill="transparent"
                  onMouseEnter={() => setHover(p)} />
          </g>
        ))}
        {c.first.base && <text x={x(c.first.base.t)} y={H - B + 13} className={s.markBase}
                               textAnchor={anchor(c.first.base.t)}>▲ base</text>}
        {c.first.model && <text x={x(c.first.model.t)} y={H - B + 13} className={s.markVar}
                                textAnchor={anchor(c.first.model.t)}>▲ new</text>}
        <line x1={x(shown.t)} x2={x(shown.t)} y1={T} y2={H - B} className={s.hoverLine} />
        <text x={L} y={H - 4} className={s.axis}>{day(points[0].time)}</text>
        <text x={W - R} y={H - 4} className={s.axis} textAnchor="end">{day(points[points.length - 1].time)}</text>
      </svg>
      <div className={s.legend}>
        <span><i className={s.keyBase} /> base</span>
        <span><i className={s.keyVar} /> base + {variantName(v)}</span>
        <span className={s.meta}>dashed: each model's cutoff · large dots: flagged rows · shaded: the rows base + new is ahead (green) or behind (amber)</span>
      </div>
      <div className={s.rowNote}>
        <span className={s.mono}>row {shown.t} · {day(shown.time)} · {shown.id}</span>
        <span>base <b>{shown.scores.base.toFixed(4)}</b>{shown.scores.base >= cut.base ? ' · flagged' : ''}</span>
        <span className={s.varInk}>base + new <b>{shown.scores[v].toFixed(4)}</b>{shown.scores[v] >= cut.model ? ' · flagged' : ''}</span>
        {model.columns.map((col) => (
          <span key={col} className={s.mono}>{col} = {shown.values[col] == null ? '–' : +Number(shown.values[col]).toFixed(4)}</span>
        ))}
      </div>
    </div>
  )
}

