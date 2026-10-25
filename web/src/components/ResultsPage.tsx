import { Fragment, useEffect, useMemo, useState } from 'react'
import { api, usePolled } from '../api'
import { navigate } from '../App'
import type { BaseModelCard, EvalResult, InferenceDetail } from '../types'
import { ConfirmButton } from './ConfirmButton'
import { EvalPanel, pct, signed } from './EvalPanel'
import { ahead, OutcomeBar, summarise, type Model } from './InferencePage'
import s from './ResultsPage.module.css'

const tone = (x: number | null | undefined) => (x == null || x === 0 ? '' : x > 0 ? s.up : s.down)
const when = (ts: number) => new Date(ts * 1000).toLocaleString([], {
  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false,
})

// What makes two rows the same variant: a feature by its run and name, a
// combination by its name and members.
const identity = (r: EvalResult) =>
  r.kind === 'feature' ? `f:${r.key ?? r.name}` : `c:${r.name}:${[...r.members].sort().join(',')}`

type SortKey = 'gini' | 'screen' | 'evaluated' | 'name' | 'missing' | 'corr' | 'vgini' | 'vgini_screen' | 'level' | `capture:${string}`

// A score level that moves more than this, either way, is marked.
const LEVEL_SHIFT = 0.05

// The route's id: an evaluation, or one variant of it - `<eval_id>.<variant>`.
export const selection = (id: string | null) => {
  if (!id) return null
  const dot = id.indexOf('.')
  return dot < 0 ? { evalId: id, variant: null } : { evalId: id.slice(0, dot), variant: id.slice(dot + 1) }
}

/**
 * Every variant evaluated: single features (base + the feature) and combinations
 * (base + every feature in it), each by its latest evaluation. A row expands to
 * what its model did on the target cases (Inference results); its details - the
 * verdict gate by gate, how the evaluation ran - open in a panel at the right.
 * Removing a row removes the variant from every evaluation.
 */
export function ResultsPage({ selected }: { selected: string | null }) {
  const [results, refreshResults] = usePolled(api.results, [], 15000)
  const [evaluations, refreshEvaluations] = usePolled(api.evaluations, [], 3000)
  const [base] = usePolled(api.baseModel, [], 15000)
  const pick = selection(selected)
  const select = (evalId: string, variant: string | null) =>
    navigate('results', variant ? `${evalId}.${variant}` : evalId)
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: 'gini', desc: true })
  const [expanded, setExpanded] = useState<string | null>(null)        // `${eval_id}:${variant}`
  const [downloading, setDownloading] = useState(false)
  // Discovery's screen figures beside the out-of-time ones - hidden unless asked for.
  const [screen, setScreen] = useState(() => {
    try { return localStorage.getItem('autofe.results.screen') === '1' } catch { return false }
  })
  const showScreen = (on: boolean) => {
    setScreen(on)
    try { localStorage.setItem('autofe.results.screen', on ? '1' : '0') } catch { /* not kept */ }
  }
  // Each evaluation's target-case scores, fetched once when a row first expands.
  const [inferences, setInferences] = useState<Record<string, InferenceDetail | 'none'>>({})
  useEffect(() => {
    const evalId = expanded?.split(':')[0]
    if (!evalId || inferences[evalId]) return
    api.inference(evalId).then((d) => setInferences((x) => ({ ...x, [evalId]: d })))
      .catch(() => setInferences((x) => ({ ...x, [evalId]: 'none' })))
  }, [expanded])  // eslint-disable-line

  const captures = useMemo(() => {
    const all = new Set((results ?? []).flatMap((r) => Object.keys(r.capture_gain)))
    return [...all].sort((a, b) => Number(b.slice(3)) - Number(a.slice(3)))
  }, [results])

  // Every evaluation of each variant - the latest is the row, all go when it is removed.
  const byVariant = useMemo(() => {
    const out = new Map<string, EvalResult[]>()
    for (const r of results ?? []) out.set(identity(r), [...(out.get(identity(r)) ?? []), r])
    return out
  }, [results])

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    const out = [...byVariant.values()].map((all) => all[0])     // newest first: [0] is the latest
      .filter((r) => !q || [r.name, r.direction, ...r.members].some((x) => x.toLowerCase().includes(q)))
    const value = (r: EvalResult): number | string => {
      if (sort.key === 'gini') return r.gini_gain ?? -Infinity
      if (sort.key === 'screen') return r.screen_gain ?? -Infinity
      if (sort.key === 'evaluated') return r.evaluated
      if (sort.key === 'name') return r.name
      if (sort.key === 'missing') return r.missing_rate ?? -Infinity
      if (sort.key === 'corr') return r.max_corr ?? -Infinity
      if (sort.key === 'vgini') return r.variable_gini ?? -Infinity
      if (sort.key === 'vgini_screen') return r.screen_variable_gini ?? -Infinity
      if (sort.key === 'level') return r.score_change == null ? -Infinity : Math.abs(r.score_change)
      return r.capture_gain[sort.key.slice(8)] ?? -Infinity
    }
    return out.sort((a, b) => {
      const x = value(a), y = value(b)
      const order = x < y ? -1 : x > y ? 1 : 0
      return sort.desc ? -order : order
    })
  }, [byVariant, query, sort])

  const Head = ({ k, label, num = true, hint }: { k: SortKey; label: string; num?: boolean; hint?: string }) => (
    <th className={num ? s.num : ''} title={hint}>
      <button className={`${s.sortBtn} ${sort.key === k ? s.sorted : ''}`}
              onClick={() => setSort({ key: k, desc: sort.key === k ? !sort.desc : k !== 'name' })}>
        {label}{sort.key === k ? (sort.desc ? ' ↓' : ' ↑') : ''}
      </button>
    </th>
  )

  const evaluated = new Set((results ?? []).map((r) => r.eval_id)).size
  const unfinished = (evaluations ?? []).filter((e) => e.status !== 'done')
  const isOn = (r: Row) => pick?.evalId === r.eval_id && pick.variant === r.variant
  const singles = rows.filter((r) => r.kind === 'feature')
  const combos = rows.filter((r) => r.kind === 'combination')
  const passed = singles.filter((r) => r.verdict === 'PASS').length

  const Evaluated = (r: Row) => (
    <td className={`${s.num} ${s.dim}`}>{when(r.evaluated)}</td>
  )
  const Gains = (r: Row) => (r.trained === false ? (
    <td colSpan={2 + captures.length} className={s.notTrained} title={r.not_trained ?? undefined}>
      {r.not_trained?.startsWith('training failed') ? 'training failed' : 'not trained - dropped before modelling'}
    </td>
  ) : (
    <>
      <td className={`${s.num} ${s.gain} ${tone(r.gini_gain)}`}>{signed(r.gini_gain)}</td>
      {captures.map((c) => (
        <td key={c} className={`${s.num} ${tone(r.capture_gain[c])}`}>{signed(r.capture_gain[c], 3)}</td>
      ))}
      {Level(r)}
    </>
  ))
  const GainHeads = () => (
    <>
      <Head k="gini" label="Test Δ Gini" />
      {captures.map((c) => <Head key={c} k={`capture:${c}`} label={`Δ capture ${c.slice(3)}%`} />)}
      <Head k="level" label="Score level change"
            hint="The mean predicted probability on test, base's → this model's, and the relative change. A new feature should not move the level much." />
    </>
  )
  const Level = (r: Row) => (
    <td className={s.num}>
      {r.score_change == null ? <span className={s.dim}>–</span> : (
        <>
          <span className={Math.abs(r.score_change) >= LEVEL_SHIFT ? s.levelShift : ''}>
            {r.score_change >= 0 ? '+' : ''}{(r.score_change * 100).toFixed(1)}%</span>
          <div className={s.corrWith}>{r.score_base?.toFixed(4)} → {r.score_mean?.toFixed(4)}</div>
        </>
      )}
    </td>
  )
  const open = (r: Row) => select(r.eval_id, r.variant)
  const rowId = (r: Row) => `${r.eval_id}:${r.variant}`
  const flip = (r: Row) => setExpanded((x) => (x === rowId(r) ? null : rowId(r)))
  const Chev = (r: Row) => <span className={s.chev}>{expanded === rowId(r) ? '▾' : '▸'}</span>
  const Expansion = (r: Row) => expanded === rowId(r) ? (
    <tr className={s.expandRow}>
      <td colSpan={100}>
        <CasesInline r={r} detail={inferences[r.eval_id] ?? null} onDetails={() => open(r)} />
      </td>
    </tr>
  ) : null
  const remove = (r: Row) => (
    <td className={s.removeCell}>
      <div className={s.removeAt}>
        <ConfirmButton label="Remove" confirm="Remove this result?" title="Take this row off the results"
                       onConfirm={async () => {
                         for (const each of byVariant.get(identity(r)) ?? [r]) {
                           await api.removeResult(each.eval_id, each.variant)
                         }
                         if (isOn(r)) navigate('results')
                         refreshResults()
                       }} />
      </div>
    </td>
  )

  return (
    <div className={s.page}>
      <header className={s.header}>
        <span className={`eyebrow ${s.eyebrow}`}>Step 3 · Evaluate</span>
        <h1 className={s.title}>Metrics results</h1>
      </header>

      <div className={`${s.body} ${pick ? s.withPanel : ''}`}>
      <div className={s.main}>
      <ModelCard card={base ?? null} />
      <section className={s.card}>
        <div className={s.toolbar}>
          <input className={s.search} value={query} placeholder="Filter by feature, combination or direction"
                 onChange={(e) => setQuery(e.target.value)} />
          <span className={s.meta}>the latest evaluation of each · from {evaluated} evaluation{evaluated === 1 ? '' : 's'}</span>
          <label className={s.screenToggle}
                 title="Discovery's screen results - Δ Gini and variable Gini on the screen rows - beside each feature, to compare with test (data shift)">
            <input type="checkbox" checked={screen} onChange={(e) => showScreen(e.target.checked)} />
            Show screen results
          </label>
          {rows.length > 0 && (
            <button className={s.download} disabled={downloading}
                    title="Every row shown - the filter applies - with its metrics and its target cases, as CSV"
                    onClick={async () => {
                      setDownloading(true)
                      try { await downloadResults([...singles, ...combos], captures) } finally { setDownloading(false) }
                    }}>
              {downloading ? 'Preparing…' : `Download CSV (${rows.length})`}
            </button>
          )}
          {(results?.length ?? 0) > 0 && (
            <ConfirmButton label="Clear all" confirm="Clear every result?"
                           title="Delete every finished evaluation - a running one stays"
                           onConfirm={async () => {
                             await api.clearResults()
                             navigate('results')
                             refreshResults()
                           }} />
          )}
        </div>

        {unfinished.length > 0 && (
          <div className={s.unfinished}>
            {unfinished.map((e) => (
              <span key={e.eval_id} className={`${s.pending} ${s[e.status]}`}>
                <button className={s.pendingOpen} onClick={() => select(e.eval_id, null)}>
                  <span className={s.pendingDot} />
                  {e.status === 'running' ? 'Evaluating' : e.status === 'stopped' ? 'Stopped' : 'Failed'}: {e.features.map((f) => f.column).join(', ')}
                  {Object.keys(e.combinations).length > 0 && ` · ⊕ ${Object.keys(e.combinations).join(', ⊕ ')}`}
                </button>
                {e.status !== 'running' && (
                  <button className={s.pendingRemove} title="Remove this evaluation"
                          aria-label="Remove this evaluation" onClick={async () => {
                            await api.deleteEvaluation(e.eval_id)
                            if (pick?.evalId === e.eval_id) navigate('results')
                            refreshEvaluations()
                          }}>×</button>
                )}
              </span>
            ))}
          </div>
        )}

        {results && results.length === 0 ? (
          <div className={s.empty}>No finished evaluations yet. Evaluate features from the pool first.</div>
        ) : (
          <div className={s.groups}>
            <div className={s.group}>
              <div className={s.groupHead}>
                <span className={s.groupTitle}>Single features</span>
                <span className={s.meta}>{singles.length} · {passed} pass · each is base + the feature</span>
              </div>
              {singles.length === 0 ? <div className={s.empty}>None match.</div> : (
                <div className={s.tableWrap}>
                  <table className={s.table}>
                    <thead>
                      <tr>
                        <Head k="name" label="Feature" num={false} />
                        {screen && (
                          <>
                            <Head k="screen" label="Δ Gini screen"
                                  hint="The Gini gain discovery's screen measured, on the screen rows - compare with Test Δ Gini for a data shift" />
                            <Head k="vgini_screen" label="Var. Gini screen"
                                  hint="The feature alone as a score for the target, |2·AUC - 1|, on the screen rows (check C2) - compare with Var. Gini OOT" />
                          </>
                        )}
                        <th>Level</th>
                        <Head k="missing" label="Missing" hint="share of rows with no value, every split" />
                        {GainHeads()}
                        <Head k="vgini" label="Var. Gini OOT"
                              hint={`The feature alone as a score for the target, |2·AUC - 1|, on test, out of time.${screen ? ' Amber: below half the screen\'s' : ''}`} />
                        <Head k="corr" label="Max |corr|" hint="largest |Spearman| with a base feature, on train - and which one" />
                        <th className={s.num}>SHAP rank</th>
                        <th className={s.num} title="The base feature it moves with most (Max |corr|): its SHAP rank in base → in base + this feature">Nearest's SHAP rank</th>
                        <th>Verdict</th>
                        <Head k="evaluated" label="Evaluated" />
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {singles.map((r) => (
                        <Fragment key={rowId(r)}>
                        <tr onClick={() => flip(r)}
                            className={isOn(r) || expanded === rowId(r) ? s.on : ''} title="Show its target cases">
                          <td>
                            <div className={s.variant}>{Chev(r)}{r.name}</div>
                            <div className={s.sub} title={r.direction}>{r.direction}</div>
                          </td>
                          {screen && (
                            <>
                              <td className={`${s.num} ${s.screenCol}`}>{signed(r.screen_gain)}</td>
                              <td className={`${s.num} ${s.screenCol}`}>
                                {r.screen_variable_gini != null ? r.screen_variable_gini.toFixed(3) : <span className={s.dim}>–</span>}</td>
                            </>
                          )}
                          <td>{r.level ? <span className={s.level}>{r.level}</span> : <span className={s.dim}>–</span>}</td>
                          <td className={s.num}>{pct(r.missing_rate)}</td>
                          {Gains(r)}
                          <td className={s.num}>
                            {r.variable_gini != null
                              ? <span className={screen && r.screen_variable_gini && r.variable_gini < r.screen_variable_gini / 2 ? s.levelShift : ''}>
                                  {r.variable_gini.toFixed(3)}</span>
                              : r.variable_gini_note
                                ? <span className={s.noSignal} title={r.variable_gini_note}>
                                    {r.variable_gini_note.startsWith('empty') ? 'empty' : 'constant'}</span>
                                : <span className={s.dim}>–</span>}
                          </td>
                          <td className={s.num}>
                            {r.max_corr != null ? r.max_corr.toFixed(3) : <span className={s.dim}>–</span>}
                            {r.max_corr_with && <div className={s.corrWith}>{r.max_corr_with}</div>}
                          </td>
                          <td className={s.num}>
                            {r.shap[0]
                              ? <span title={`${(r.shap[0].share * 100).toFixed(1)}% of SHAP`}>
                                  <b>#{r.shap[0].rank}</b><span className={s.dim}> of {r.shap[0].of}</span></span>
                              : <span className={s.dim}>–</span>}
                          </td>
                          <td className={s.num}>
                            {r.nearest_shap && (r.nearest_shap.base != null || r.nearest_shap.variant != null)
                              ? <span title={r.nearest_shap.feature}>
                                  #{r.nearest_shap.base ?? '–'} → <b>#{r.nearest_shap.variant ?? '–'}</b></span>
                              : <span className={s.dim}>–</span>}
                          </td>
                          <td>
                            {r.verdict
                              ? <span className={r.verdict === 'PASS' ? s.pass : s.fail}
                                      title={r.reason || undefined}>{r.verdict}</span>
                              : <span className={s.dim}>–</span>}
                          </td>
                          {Evaluated(r)}
                          {remove(r)}
                        </tr>
                        {Expansion(r)}
                        </Fragment>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div className={s.group}>
              <div className={s.groupHead}>
                <span className={s.groupTitle}>Combinations</span>
                <span className={s.meta}>{combos.length} · each is base + every feature in it, in one model</span>
              </div>
              {/* The latest evaluation that asked for base + its singles together, if that model was not trained. */}
              {(() => {
                const asked = (evaluations ?? []).find((e) => e.status === 'done' && e.together)
                return asked?.together_note ? (
                  <div className={s.togetherNote}>
                    ⊕ {asked.together === 'all' ? 'all new features' : 'every passing feature'} (evaluation of {when(asked.started)}):
                    {' '}{asked.together_note.replace(/^base \+ every [a-z ]+: /, '')}
                  </div>
                ) : null
              })()}
              {combos.length === 0 ? <div className={s.empty}>None yet - combine features on the Evaluate page.</div> : (
                <div className={s.tableWrap}>
                  <table className={s.table}>
                    <thead>
                      <tr>
                        <Head k="name" label="Combination" num={false} />
                        {GainHeads()}
                        <th>SHAP rank of each feature in the combined model</th>
                        <Head k="evaluated" label="Evaluated" />
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {combos.map((r) => (
                        <Fragment key={rowId(r)}>
                        <tr onClick={() => flip(r)}
                            className={isOn(r) || expanded === rowId(r) ? s.on : ''} title="Show its target cases">
                          <td>
                            <div className={s.variant}>{Chev(r)}⊕ {r.auto ? (r.name === 'all_passing' ? 'every passing feature' : 'all new features') : r.name}</div>
                            <div className={s.sub}>{r.members.length} features{r.auto ? ' · added by the evaluation' : ''}</div>
                          </td>
                          {Gains(r)}
                          <td>
                            <div className={s.members}>
                              {memberRanks(r).map((m) => (
                                <div key={m.feature} className={s.member}
                                     title={m.share != null ? `${(m.share * 100).toFixed(1)}% of SHAP` : 'no SHAP rank recorded'}>
                                  <span className={s.memberName}>{m.feature}</span>
                                  {m.rank != null
                                    ? <span className={s.num}><b>#{m.rank}</b><span className={s.dim}> of {m.of}</span></span>
                                    : <span className={s.dim}>–</span>}
                                </div>
                              ))}
                            </div>
                          </td>
                          {Evaluated(r)}
                          {remove(r)}
                        </tr>
                        {Expansion(r)}
                        </Fragment>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </section>
      </div>
      {pick && (
        <EvalPanel key={pick.evalId} evalId={pick.evalId} variant={pick.variant}
                   onSelect={(v) => select(pick.evalId, v)}
                   onClose={() => navigate('results')}
                   onDeleted={() => { refreshResults(); navigate('results') }} />
      )}
      </div>
    </div>
  )
}

type Row = EvalResult

/** The results as shown - every row, its metrics, and what its model did on the
 *  target cases (each model flagging its own top share of test, as on Inference
 *  results) - as a CSV file. */
async function downloadResults(rows: Row[], captures: string[]) {
  const details: Record<string, InferenceDetail | null> = {}
  await Promise.all([...new Set(rows.map((r) => r.eval_id))].map((id) =>
    api.inference(id).then((d) => { details[id] = d }).catch(() => { details[id] = null })))
  const cases = (r: Row) => {
    const d = details[r.eval_id]
    if (!d?.done?.summaries[r.variant]) return null
    const sum = summarise({ evalId: r.eval_id, evaluated: r.evaluated, variant: r.variant,
                            columns: r.members, gini: r.gini_gain, detail: d },
                          { mode: 'quantile', share: d.flag_top, score: 0.1 })
    const n = (o: string) => sum.counts.find(([x]) => x === o)?.[1] ?? 0
    return { share: d.flag_top, total: sum.rows.length, sooner: n('sooner'), same: n('same'),
             later: n('later'), neither: n('neither'), base: sum.base, model: sum.model, rows: sum.total }
  }
  // feature_id (member_feature_ids for a combination) is the logging board's
  // feature_id: <run_id>:<name>, the same feature in both downloads.
  const head = ['feature_id', 'member_feature_ids', 'kind', 'name', 'members', 'direction', 'level',
    'run_id', 'eval_id', 'evaluated', 'test_gini_gain',
    ...captures.map((c) => `capture_gain_${c}_test`),
    'screen_gini_gain', 'variable_gini_screen', 'variable_gini_oot', 'variable_gini_oot_note',
    'missing_rate', 'max_corr', 'max_corr_with',
    'score_mean_base', 'score_mean_model', 'score_level_change', 'shap_rank', 'shap_of', 'shap_share',
    'nearest_feature', 'nearest_shap_rank_base', 'nearest_shap_rank_model', 'verdict', 'reason',
    'trained', 'not_trained', 'cases_flag_top_share', 'cases', 'cases_sooner', 'cases_same',
    'cases_later', 'cases_neither', 'cases_flagged_base', 'cases_flagged_model', 'cases_total_rows_earlier']
  const lines = rows.map((r) => {
    const c = cases(r)
    return [r.kind === 'feature' ? r.key : '', r.kind === 'combination' ? (r.member_keys ?? []).join(' | ') : '',
      r.kind, r.auto ? (r.name === 'all_passing' ? 'every passing feature' : 'all new features') : r.name,
      r.members.join(' + '), r.direction, r.level, r.run_id, r.eval_id,
      new Date(r.evaluated * 1000).toISOString(), r.gini_gain,
      ...captures.map((p) => r.capture_gain[p]),
      r.screen_gain, r.screen_variable_gini, r.variable_gini, r.variable_gini_note,
      r.missing_rate, r.max_corr, r.max_corr_with,
      r.score_base, r.score_mean, r.score_change, r.shap[0]?.rank, r.shap[0]?.of, r.shap[0]?.share,
      r.nearest_shap?.feature, r.nearest_shap?.base, r.nearest_shap?.variant, r.verdict, r.reason,
      r.trained, r.not_trained, c?.share, c?.total, c?.sooner, c?.same, c?.later, c?.neither,
      c?.base, c?.model, c?.rows]
  })
  const cell = (v: unknown) => {
    const text = v === null || v === undefined ? '' : String(v)
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
  }
  const csv = [head, ...lines].map((l) => l.map(cell).join(',')).join('\r\n')
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `evaluation_results_${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

/** A metrics row's model on the target cases: how its flags move against base's,
 *  at each model's own top share of test - the Inference results page's first rule. */
function CasesInline({ r, detail, onDetails }: {
  r: Row; detail: InferenceDetail | 'none' | null; onDetails: () => void
}) {
  const actions = (
    <div className={s.casesActions}>
      <button className={s.casesBtn} onClick={onDetails}>Training details ›</button>
      {detail && detail !== 'none' && detail.done?.summaries[r.variant] && (
        <button className={s.casesBtn} onClick={() => navigate('inference', `${r.eval_id}.${r.variant}`)}>
          Open in Inference results ›</button>
      )}
    </div>
  )
  let body: React.ReactNode
  if (detail === null) body = <span className={s.dim}>Loading the target cases…</span>
  else if (detail === 'none') body = <span className={s.dim}>The target cases were not scored with this evaluation - set them up in Setup, then score them on Inference results.</span>
  else if (detail.status === 'running') body = <span className={s.dim}>Scoring the target cases…</span>
  else if (!detail.done) body = <span className={s.dim}>Scoring the target cases {detail.status}{detail.error ? `: ${detail.error}` : ''}.</span>
  else if (!detail.done.summaries[r.variant]) body = <span className={s.dim}>This model was not scored on the target cases.</span>
  else {
    const model: Model = { evalId: r.eval_id, evaluated: r.evaluated, variant: r.variant, columns: r.members,
                           gini: r.gini_gain, detail }
    const rule = { mode: 'quantile' as const, share: detail.flag_top, score: 0.1 }
    const sum = summarise(model, rule)
    const sooner = sum.rows.filter((c) => c.outcome === 'sooner')
      .sort((a, b) => (b.rowsSooner ?? 0) - (a.rowsSooner ?? 0)).slice(0, 4)
    body = !sum.cut ? <span className={s.dim}>No flag cutoff for this model.</span> : (
      <div className={s.cases}>
        <div className={s.casesHead}>
          <b>Target cases</b>
          <span className={s.dim}>{sum.rows.length} cases · each model flags its own top {+(rule.share * 100).toFixed(1)}% of test scores</span>
        </div>
        <div className={s.casesGrid}>
          <OutcomeBar counts={sum.counts} total={sum.rows.length} />
          <div className={s.casesFact}><span className={s.dim}>Cases flagged</span>
            <b>{sum.base} → <span className={sum.model > sum.base ? s.up : sum.model < sum.base ? s.down : ''}>{sum.model}</span></b></div>
          <div className={s.casesFact}><span className={s.dim}>Total rows earlier</span>
            <b className={sum.total != null ? s.up : ''}>{sum.total ?? '–'}</b></div>
        </div>
        {sooner.length > 0 && (
          <div className={s.casesSooner}>
            <span className={s.dim}>Flagged sooner:</span>
            {sooner.map((c) => <span key={c.case} className={s.caseChip}><code>{c.case}</code> {ahead(c)}</span>)}
          </div>
        )}
      </div>
    )
  }
  return <div className={s.casesWrap} onClick={(e) => e.stopPropagation()}>{body}{actions}</div>
}

/** Every member of a combination with its rank in that model - best first, unranked last. */
function memberRanks(r: EvalResult) {
  const ranked = new Map(r.shap.map((k) => [k.feature, k]))
  return r.members
    .map((feature) => ({ feature, rank: ranked.get(feature)?.rank ?? null,
                         of: ranked.get(feature)?.of ?? null, share: ranked.get(feature)?.share ?? null }))
    .sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity))
}

/** The base model every row is measured against: how it scores, on what, and what drives it. */
function ModelCard({ card }: { card: BaseModelCard | null }) {
  const [open, setOpen] = useState(true)
  if (!card) {
    return (
      <section className={s.model}>
        <div className={s.modelHead}><span className={s.modelTitle}>Base model</span>
          <span className={s.meta}>Trained with the first evaluation - its card shows here then.</span></div>
      </section>
    )
  }
  const { splits } = card
  const top = card.drivers[0]?.share || 1
  const pctOf = (x: number | null | undefined) => (x == null ? '–' : `${(x * 100).toFixed(1)}%`)
  const rows: { label: string; values: (string)[] }[] = [
    { label: 'Gini', values: splits.map((sp) => card.gini[sp]?.toFixed(4) ?? '–') },
    ...Object.entries(card.capture).map(([k, by]) => ({
      label: `Capture top ${k.slice(3)}%`, values: splits.map((sp) => pctOf(by[sp])) })),
    { label: 'Rows', values: splits.map((sp) => card.data[sp]?.rows.toLocaleString() ?? '–') },
    { label: `Default rate`, values: splits.map((sp) => pctOf(card.data[sp]?.default_rate)) },
  ]
  return (
    <section className={s.model}>
      <button className={s.modelHead} onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className={s.chev}>{open ? '▾' : '▸'}</span>
        <span className={s.modelTitle}>Base model</span>
        <span className={s.modelFacts}>
          XGBoost · <b>{card.n_features.toLocaleString()}</b> features · test Gini <b>{card.gini.test?.toFixed(4) ?? '–'}</b>
        </span>
        <span className={s.meta}>target <code>{card.target}</code> · from the evaluation of {when(card.evaluated)}</span>
      </button>
      {open && (
        <div className={s.modelBody}>
          <table className={s.modelTable}>
            <thead>
              <tr><th />{splits.map((sp) => <th key={sp} className={`${s.num} ${sp === 'test' ? s.testCol : ''}`}>{sp}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.label}>
                  <td>{r.label}</td>
                  {r.values.map((v, i) => (
                    <td key={splits[i]} className={`${s.num} ${splits[i] === 'test' ? s.testCol : ''}`}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div className={s.drivers}>
            <span className={s.driversTitle}>Top {card.drivers.length} of {card.n_features.toLocaleString()} features · SHAP share on test</span>
            <div className={s.driverList}>
              {card.drivers.map((d, i) => (
                <div key={d.feature} className={s.driver}>
                  <span className={s.driverRank}>{i + 1}</span>
                  <span className={s.driverName} title={d.feature}>{d.feature}</span>
                  <span className={s.driverBar}><i style={{ width: `${(d.share / top) * 100}%` }} /></span>
                  <span className={s.driverPct}>{pctOf(d.share)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
