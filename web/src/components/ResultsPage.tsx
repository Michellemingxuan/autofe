import { useMemo, useState } from 'react'
import { api, usePolled } from '../api'
import { navigate } from '../App'
import type { EvalResult } from '../types'
import { ConfirmButton } from './ConfirmButton'
import { EvalPanel, pct, signed } from './EvalPanel'
import s from './ResultsPage.module.css'

const tone = (x: number | null | undefined) => (x == null || x === 0 ? '' : x > 0 ? s.up : s.down)
const when = (ts: number) => new Date(ts * 1000).toLocaleString([], {
  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false,
})

// What makes two rows the same variant: a feature by its run and name, a
// combination by its name and members.
const identity = (r: EvalResult) =>
  r.kind === 'feature' ? `f:${r.key ?? r.name}` : `c:${r.name}:${[...r.members].sort().join(',')}`

type SortKey = 'gini' | 'screen' | 'evaluated' | 'name' | 'missing' | 'corr' | `capture:${string}`

// The route's id: an evaluation, or one variant of it - `<eval_id>.<variant>`.
export const selection = (id: string | null) => {
  if (!id) return null
  const dot = id.indexOf('.')
  return dot < 0 ? { evalId: id, variant: null } : { evalId: id.slice(0, dot), variant: id.slice(dot + 1) }
}

/**
 * Every variant evaluated: single features (base + the feature) and combinations
 * (base + every feature in it), each by its latest evaluation. A row opens its
 * details - the verdict gate by gate, the feature, how the evaluation ran - in a
 * panel at the right. Removing a row removes the variant from every evaluation.
 */
export function ResultsPage({ selected }: { selected: string | null }) {
  const [results, refreshResults] = usePolled(api.results, [], 15000)
  const [evaluations] = usePolled(api.evaluations, [], 3000)
  const pick = selection(selected)
  const select = (evalId: string, variant: string | null) =>
    navigate('results', variant ? `${evalId}.${variant}` : evalId)
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: 'gini', desc: true })

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

  const Evaluated = ({ r }: { r: Row }) => (
    <td className={`${s.num} ${s.dim}`}>{when(r.evaluated)}</td>
  )
  const Gains = ({ r }: { r: Row }) => (
    <>
      <td className={`${s.num} ${s.gain} ${tone(r.gini_gain)}`}>{signed(r.gini_gain)}</td>
      {captures.map((c) => (
        <td key={c} className={`${s.num} ${tone(r.capture_gain[c])}`}>{signed(r.capture_gain[c], 3)}</td>
      ))}
    </>
  )
  const GainHeads = () => (
    <>
      <Head k="gini" label="Test Δ Gini" />
      {captures.map((c) => <Head key={c} k={`capture:${c}`} label={`Δ capture ${c.slice(3)}%`} />)}
    </>
  )
  const open = (r: Row) => select(r.eval_id, r.variant)
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
        <h1 className={s.title}>All evaluation results</h1>
      </header>

      <div className={`${s.body} ${pick ? s.withPanel : ''}`}>
      <section className={s.card}>
        <div className={s.toolbar}>
          <input className={s.search} value={query} placeholder="Filter by feature, combination or direction"
                 onChange={(e) => setQuery(e.target.value)} />
          <span className={s.meta}>the latest evaluation of each · from {evaluated} evaluation{evaluated === 1 ? '' : 's'}</span>
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
              <button key={e.eval_id} className={`${s.pending} ${s[e.status]}`}
                      onClick={() => select(e.eval_id, null)}>
                <span className={s.pendingDot} />
                {e.status === 'running' ? 'Evaluating' : e.status === 'stopped' ? 'Stopped' : 'Failed'}: {e.features.map((f) => f.column).join(', ')}
                {Object.keys(e.combinations).length > 0 && ` · ⊕ ${Object.keys(e.combinations).join(', ⊕ ')}`}
              </button>
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
                        <th>Level</th>
                        <Head k="screen" label="Screen Δ Gini" />
                        <Head k="missing" label="Missing" hint="share of rows with no value, every split" />
                        <Head k="corr" label="Max |corr|" hint="largest |Spearman| with a base feature, on train - and which one" />
                        <GainHeads />
                        <th className={s.num}>SHAP rank</th>
                        <th>Verdict</th>
                        <Head k="evaluated" label="Evaluated" />
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {singles.map((r) => (
                        <tr key={`${r.eval_id}:${r.variant}`} onClick={() => open(r)}
                            className={isOn(r) ? s.on : ''} title="Show the details">
                          <td>
                            <div className={s.variant}>{r.name}</div>
                            <div className={s.sub} title={r.direction}>{r.direction}</div>
                          </td>
                          <td>{r.level ? <span className={s.level}>{r.level}</span> : <span className={s.dim}>–</span>}</td>
                          <td className={`${s.num} ${s.dim}`}>{signed(r.screen_gain)}</td>
                          <td className={s.num}>{pct(r.missing_rate)}</td>
                          <td className={s.num}>
                            {r.max_corr != null ? r.max_corr.toFixed(3) : <span className={s.dim}>–</span>}
                            {r.max_corr_with && <div className={s.corrWith}>{r.max_corr_with}</div>}
                          </td>
                          <Gains r={r} />
                          <td className={s.num}>
                            {r.shap[0]
                              ? <span title={`${(r.shap[0].share * 100).toFixed(1)}% of SHAP`}>
                                  <b>#{r.shap[0].rank}</b><span className={s.dim}> of {r.shap[0].of}</span></span>
                              : <span className={s.dim}>–</span>}
                          </td>
                          <td>
                            {r.verdict
                              ? <span className={r.verdict === 'PASS' ? s.pass : s.fail}
                                      title={r.reason || undefined}>{r.verdict}</span>
                              : <span className={s.dim}>–</span>}
                          </td>
                          <Evaluated r={r} />
                          {remove(r)}
                        </tr>
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
              {combos.length === 0 ? <div className={s.empty}>None yet - combine features on the Evaluate page.</div> : (
                <div className={s.tableWrap}>
                  <table className={s.table}>
                    <thead>
                      <tr>
                        <Head k="name" label="Combination" num={false} />
                        <GainHeads />
                        <th>SHAP rank of each feature in the combined model</th>
                        <Head k="evaluated" label="Evaluated" />
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {combos.map((r) => (
                        <tr key={`${r.eval_id}:${r.variant}`} onClick={() => open(r)}
                            className={isOn(r) ? s.on : ''} title="Show the details">
                          <td>
                            <div className={s.variant}>⊕ {r.name}</div>
                            <div className={s.sub}>{r.members.length} features</div>
                          </td>
                          <Gains r={r} />
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
                          <Evaluated r={r} />
                          {remove(r)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
        <div className={s.hint}>
          Gains are against base on test, the out-of-time hold-out. Screen Δ Gini is the gain the
          feature showed in discovery, on the screen rows. Capture = share of defaults in the top x%
          of scores. Missing is the share of rows with no value, over every split; Max |corr| is
          the largest |Spearman| with a base feature, on train. Combinations have no verdict: the
          gates judge single features.
        </div>
      </section>
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

/** Every member of a combination with its rank in that model - best first, unranked last. */
function memberRanks(r: EvalResult) {
  const ranked = new Map(r.shap.map((k) => [k.feature, k]))
  return r.members
    .map((feature) => ({ feature, rank: ranked.get(feature)?.rank ?? null,
                         of: ranked.get(feature)?.of ?? null, share: ranked.get(feature)?.share ?? null }))
    .sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity))
}
