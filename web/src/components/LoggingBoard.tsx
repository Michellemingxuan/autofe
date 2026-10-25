import { Fragment, useEffect, useMemo, useState } from 'react'
import { api, usePolled } from '../api'
import type { BoardRow } from '../types'
import s from './LoggingBoard.module.css'
import { BoardFilters, applies, type Rule } from './BoardFilters'

type Props = { onClose: () => void }

const ENGINE: Record<string, string> = {
  mechanism: 'Mechanism', contrastive: 'Contrastive', analogy: 'Analogy',
  morphological: 'Morphological', others: 'Others',
}

const STATUS_HINT: Record<string, string> = {
  active: 'Carried signal: a feature passed the checks',
  explore: 'Nothing measured yet',
  low: 'Measured, no signal yet - Thompson sampling works it later',
}

const num = (v: number | null, digits = 3, signed = false) =>
  v === null || v === undefined ? '' : `${signed && v > 0 ? '+' : ''}${v.toFixed(digits)}`

const passed = (r: BoardRow) => r.status === 'verified' || r.status === 'kept'


// ── Columns: what each sorts by ─────────────────────────────────────────
type Key = 'direction' | 'engine' | 'idea' | 'idea_status' | 'name' | 'level' | 'seen_before'
  | 'max_model_rho' | 'variable_gini' | 'gini_gain' | 'status' | 'sent'

const COLUMNS: { key: Key; label: string; hint?: string; numeric?: boolean }[] = [
  { key: 'direction', label: 'Direction' },
  { key: 'engine', label: 'Engine' },
  { key: 'idea', label: 'Idea' },
  { key: 'idea_status', label: 'Idea status', hint: 'Sorts by P(useful): the mean of the idea\'s Beta(1 + signal, 1 + none)', numeric: true },
  { key: 'name', label: 'Feature / request' },
  { key: 'level', label: 'Level' },
  { key: 'seen_before', label: 'C0 (seen before?)', hint: 'Against every earlier feature and the model history: the same code or values are refused; a close meaning needs `differs`. Shows the closest earlier feature, or what it resembles' },
  { key: 'max_model_rho', label: 'C1 (redundancy)', hint: 'The largest |Spearman rho| with any one model column, and that column. Above 0.9: a near-copy - unless it tracks what the model misses (|ρ| with its residual ≥ 0.05)', numeric: true },
  { key: 'variable_gini', label: 'C2 (variable Gini)', hint: 'Information only: the Gini of the feature alone as a score for the outcome, |2·AUC − 1| on the screen rows', numeric: true },
  { key: 'gini_gain', label: 'C3 (screen gain)', hint: 'The screen\'s Gini gain, base vs base + the feature - against your gates', numeric: true },
  { key: 'status', label: 'Status', hint: 'Feature: verified (passed C0-C1 and the screen gates) or failed (the reason says where: script, C1, guards, gates). Request: kept, or dropped - by its challenge (the current data supplies it) or by the analyst. Idea alone: not tried' },
  { key: 'sent', label: 'C4 (evaluation)', hint: 'Verified features only - whether it was evaluated: not evaluated · evaluated · evaluated · removed (the analyst took it off) · removed (taken off before any evaluation). Its results are on Metrics results' },
]
const IDEA_COLS = 4                       // direction, engine, idea, idea status: merged per idea
const N_COLS = COLUMNS.length + 1         // + code

const EVAL_CLASS: Record<string, string> = {
  'evaluated': 'evaluated', 'evaluated · removed': 'removed', 'removed': 'removed',
  'not evaluated': 'notEvaluated',
}

const SENT_ORDER = ['evaluated', 'evaluated · removed', 'not evaluated', 'removed']

function sortValue(r: BoardRow, key: Key): string | number | null {
  switch (key) {
    case 'idea_status': return r.p_useful
    case 'variable_gini': return r.variable_gini ?? null
    case 'sent': return r.sent ? SENT_ORDER.indexOf(r.sent) : null
    case 'status': return r.kind === 'idea' ? null : r.status
    default: {
      const v = r[key]
      return v === '' || v === undefined ? null : (v as string | number | null)
    }
  }
}

function compare(a: BoardRow, b: BoardRow, key: Key, dir: 1 | -1): number {
  const x = sortValue(a, key), y = sortValue(b, key)
  if (x === null && y === null) return 0
  if (x === null) return 1                // empty values last, either way
  if (y === null) return -1
  return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y))) * dir
}

// ── Download ────────────────────────────────────────────────────────────
// The columns of the download, in order - every field the board holds, code included.
const CSV_COLUMNS: (keyof BoardRow)[] = [
  'feature_id', 'run_id', 'direction', 'round', 'engine', 'lens', 'idea', 'idea_status', 'p_useful',
  'idea_description', 'idea_measures', 'kind', 'name', 'level', 'description', 'differs', 'attempt',
  'seen_before', 'max_model_rho', 'nearest', 'nearest_rho', 'residual_rho', 'variable_gini', 'note', 'gini_gain', 'capture_gain',
  'status', 'reason', 'evidence', 'sent', 'code',
]

function toCsv(rows: BoardRow[]): string {
  const cell = (v: unknown) => {
    const text = v === null || v === undefined ? '' : Array.isArray(v) ? v.join(' | ') : String(v)
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
  }
  return [CSV_COLUMNS.join(','), ...rows.map((r) => CSV_COLUMNS.map((c) => cell(r[c])).join(','))]
    .join('\r\n')
}

function download(rows: BoardRow[]) {
  const blob = new Blob([toCsv(rows)], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `logging_board_${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

const searchText = (r: BoardRow) => [
  r.direction, r.run_id, r.engine, r.lens, r.idea, r.idea_description, ...(r.idea_measures ?? []), r.idea_status, r.name,
  r.description, r.differs, r.nearest, r.status, r.reason, r.sent,
].join(' ').toLowerCase()

const rowId = (r: BoardRow) => `${r.run_id}:${r.kind}:${r.name}:${r.idea}:${r.attempt}`
const ideaKey = (r: BoardRow) => `${r.run_id}:${r.idea}`

/** What the board's rows, statuses and columns mean - opened with the ⓘ button. */
function Schema() {
  return (
    <div className={s.schema}>
      <div>
        <h4>Rows</h4>
        <dl>
          <dt>feature</dt><dd>A feature the agent built and screened (L1 / L2).</dd>
          <dt>request</dt><dd>A data request (L3): SQL within a scope, or data beyond every scope.</dd>
          <dt>idea</dt><dd>An idea nothing was built from yet. Ideas with features show beside them.</dd>
        </dl>
      </div>
      <div>
        <h4>Status</h4>
        <dl>
          <dt>feature – verified</dt><dd>Passed C0 and C1, and cleared the screen gates (Gini gain, and capture gain if set).</dd>
          <dt>feature – failed</dt><dd>Built, but did not pass: the reason says where - the script, C1, a screen guard (redundant, spikes, infinite values) or the gates.</dd>
          <dt>request – kept</dt><dd>The challenge found the current data cannot supply it (or the analyst approved it, in older runs).</dd>
          <dt>request – dropped</dt><dd>Dropped by its challenge (the current data already supplies it), by the analyst, or never challenged before its run ended.</dd>
          <dt>request – pending</dt><dd>Waiting for its challenge, in a run still going.</dd>
          <dt>idea – not tried</dt><dd>Nothing built from it yet.</dd>
        </dl>
        <p>A feature refused by C0 - the same code as an earlier one, or the same meaning with no word on what differs - has no row: nothing was built, nothing spent.</p>
      </div>
      <div>
        <h4>Evaluation <span>(verified features only)</span></h4>
        <dl>
          <dt>not evaluated</dt><dd>In the feature set, not evaluated yet.</dd>
          <dt>evaluated</dt><dd>Measured on the full splits. Its results - verdict, out-of-time Gini - are on Metrics results, under the same feature id; the board stays with what discovery saw.</dd>
          <dt>evaluated · removed</dt><dd>Evaluated, then taken off by the analyst.</dd>
          <dt>removed</dt><dd>Taken off the feature set before any evaluation.</dd>
        </dl>
      </div>
      <div>
        <h4>Idea</h4>
        <dl>
          <dt>Engine</dt><dd>How the idea was made: mechanism, contrastive, analogy, morphological, or others.</dd>
          <dt>Idea status</dt><dd>explore - nothing measured · active - a feature carried signal · low - measured, no signal yet. Low ideas are worked later (Thompson sampling), never refused.</dd>
          <dt>P(useful)</dt><dd>The mean of Beta(1 + features with signal, 1 + features without).</dd>
        </dl>
      </div>
      <div>
        <h4>Checks</h4>
        <dl>
          <dt>C0 (seen before?)</dt><dd>Against every earlier feature and the model history: the same code or values are refused; a close meaning needs `differs`. Shows the closest earlier feature, or what it resembles.</dd>
          <dt>C1 (redundancy)</dt><dd>The largest |Spearman ρ| with any one model column, and that column. Fails above 0.9, unless its |ρ| with the base model's residual (y − ŷ) reaches 0.05.</dd>
          <dt>C2 (variable Gini)</dt><dd>The Gini of the feature alone as a score for the outcome, |2·AUC − 1| on the screen rows. Information only - set agent.checks.min_variable_gini to make it a gate.</dd>
          <dt>C3 (screen gain)</dt><dd>The screen's Gini gain, base vs base + the feature, against your gates.</dd>
          <dt>C4 (evaluation)</dt><dd>Whether it was evaluated - see Evaluation.</dd>
        </dl>
      </div>
    </div>
  )
}

/** The proposal board: every idea, what was built from it, what the checks
 *  measured, and whether it reached evaluation - opened from the rail. One row
 *  per feature or request; sort by any column, search by keyword. */
export function LoggingBoard({ onClose }: Props) {
  const [rows] = usePolled(api.board, [], 5000)
  const [rules, setRules] = useState<Rule[]>([])     // the user's filters (BoardFilters)
  const [schema, setSchema] = useState(false)
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<{ key: Key; dir: 1 | -1 } | null>(null)
  const [open, setOpen] = useState<Record<string, boolean>>({})

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // What the filters and the search show, in the chosen order (the server's order
  // when none: newest run first, ideas as recorded) - and what the download takes.
  const visible = useMemo(() => {
    const words = query.toLowerCase().split(/\s+/).filter(Boolean)
    const kept = (rows ?? []).filter((r) => applies(rules, r)
      && words.every((w) => searchText(r).includes(w)))
    return sort ? [...kept].sort((a, b) => compare(a, b, sort.key, sort.dir)) : kept
  }, [rows, rules, query, sort])

  // The statistics of what is shown.
  const stats = useMemo(() => {
    const made = visible.filter((r) => r.kind !== 'idea')
    return {
      ideas: new Set(visible.filter((r) => r.idea).map(ideaKey)).size,
      features: made.length,
      validated: made.filter(passed).length,
    }
  }, [visible])

  // First click: a sensible direction (numbers high first, words A-Z); second:
  // the other way; third: back to the order the runs recorded.
  const sortBy = (key: Key) => setSort((cur) => {
    const start: 1 | -1 = COLUMNS.find((c) => c.key === key)?.numeric ? -1 : 1
    if (!cur || cur.key !== key) return { key, dir: start }
    return cur.dir === start ? { key, dir: start === 1 ? -1 : 1 } : null
  })

  return (
    <div className={s.backdrop} onClick={onClose}>
      <section className={s.panel} onClick={(e) => e.stopPropagation()}
               role="dialog" aria-label="Logging board">
        <header className={s.head}>
          <div className={s.stats}>
            <span className="eyebrow">Logging board</span>
            <div className={s.stat}><b>{stats.ideas}</b> ideas</div>
            <div className={s.stat}><b>{stats.features}</b> features</div>
            <div className={s.stat} title="Verified features and kept data requests">
              <b>{stats.validated}</b> validated
            </div>
          </div>
          <button className={s.close} onClick={onClose} aria-label="Close">×</button>
        </header>

        <div className={s.filters}>
          <BoardFilters rows={rows ?? []} rules={rules} onChange={setRules} />
          <input className={s.search} type="search" value={query} placeholder="Search ideas and features…"
                 onChange={(e) => setQuery(e.target.value)}
                 title="Every word must appear in the idea, the feature, the direction, the engine or the status" />
          {sort && (
            <button className={`${s.chip} ${s.sortChip}`} onClick={() => setSort(null)} title="Back to the order the runs recorded">
              sorted by {COLUMNS.find((c) => c.key === sort.key)?.label} {sort.dir === 1 ? '↑' : '↓'} ×
            </button>
          )}
          <button className={`${s.chip} ${schema ? s.chipOn : ''}`} onClick={() => setSchema((v) => !v)}
                  title="What every kind, status and column means">ⓘ Schema</button>
          <button className={s.download} disabled={!visible.length} onClick={() => download(visible)}
                  title="Every row the filters show, in this order, with its code - as CSV">
            Download CSV ({visible.length})
          </button>
        </div>

        {schema && <Schema />}

        <div className={s.scroll}>
          {rows === null && <div className={s.empty}>Loading…</div>}
          {rows !== null && visible.length === 0 && (
            <div className={s.empty}>Nothing here yet: ideas appear once a direction records them.</div>
          )}
          {visible.length > 0 && (
            <table className={s.table}>
              <thead>
                <tr>
                  {COLUMNS.map((c) => (
                    <th key={c.key} title={c.hint ?? `Sort by ${c.label.toLowerCase()}`}
                        className={`${s.sortable} ${sort?.key === c.key ? s.sorted : ''}`}
                        onClick={() => sortBy(c.key)}>
                      {c.label}
                      <span className={s.arrow}>{sort?.key === c.key ? (sort.dir === 1 ? '↑' : '↓') : '↕'}</span>
                    </th>
                  ))}
                  <th>Code</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((r, i) => {
                      // An idea's cells are merged over its consecutive rows (and their open code).
                      const first = i === 0 || ideaKey(visible[i - 1]) !== ideaKey(r)
                      let span = 0
                      if (first) {
                        for (let j = i; j < visible.length && ideaKey(visible[j]) === ideaKey(r); j++) {
                          span += 1 + (open[rowId(visible[j])] ? 1 : 0)
                        }
                      }
                      const id = rowId(r)
                      const failed = r.status === 'failed' || r.status === 'dropped'
                      return (
                        <Fragment key={id}>
                          <tr className={`${s.row} ${first ? s.firstOfIdea : ''}`}>
                            {first && (
                              <>
                                <td rowSpan={span} className={s.ideaCell}>
                                  <div className={s.direction} title={r.direction}>{r.direction || 'Open exploration'}</div>
                                  <div className={s.runId}>{r.run_id}</div>
                                </td>
                                <td rowSpan={span} className={s.ideaCell}>
                                  {r.engine
                                    ? <span className={`${s.engine} ${s[r.engine]}`}>{ENGINE[r.engine] ?? r.engine}</span>
                                    : <span className={s.none} title="The idea was recorded before ideas named their engine">not recorded</span>}
                                </td>
                                <td rowSpan={span} className={s.ideaCell}>
                                  <div className={s.ideaName}>{r.idea || '(no recorded idea)'}</div>
                                  <div className={s.ideaMeta}>
                                    {r.lens && <span>{r.lens}</span>}
                                    {r.round !== null && r.round !== undefined && <span>round {r.round}</span>}
                                  </div>
                                  {r.idea_description && <div className={s.ideaDesc}>{r.idea_description}</div>}
                                  {r.idea_measures?.length > 0 && (
                                    <ul className={s.measures} title="Ways to measure the idea">
                                      {r.idea_measures.map((m) => <li key={m}>{m}</li>)}
                                    </ul>
                                  )}
                                </td>
                                <td rowSpan={span} className={s.ideaCell}>
                                  {r.idea_status && (
                                    <span className={`${s.status} ${s[r.idea_status]}`} title={STATUS_HINT[r.idea_status]}>
                                      {r.idea_status}
                                    </span>
                                  )}
                                  {r.p_useful !== null && (
                                    <div className={s.p} title="P(useful): the mean of the idea's Beta(1 + signal, 1 + none)">
                                      <span className={s.pBar}><span style={{ width: `${r.p_useful * 100}%` }} /></span>
                                      {r.p_useful.toFixed(2)}
                                    </div>
                                  )}
                                </td>
                              </>
                            )}
                            {r.kind === 'idea' ? (
                              <td colSpan={N_COLS - IDEA_COLS} className={s.none}>Not tried yet</td>
                            ) : (
                              <>
                                <td>
                                  <div className={s.name}>{r.name}</div>
                                  {r.description && <div className={s.desc}>{r.description}</div>}
                                  {r.differs && <div className={s.differs}>differs: {r.differs}</div>}
                                </td>
                                <td><span className={s.level}>{r.level}</span></td>
                                <td className={s.check}>{r.seen_before}</td>
                                <td className={`${s.mono} ${failed && r.reason.startsWith('explained') ? s.bad : ''}`}
                                    title={r.nearest ? `Spearman rho with the model column ${r.nearest}` : undefined}>
                                  {r.nearest && (
                                    <>ρ {num(r.nearest_rho ?? r.max_model_rho, 2, true)} <span className={s.nearest}>· {r.nearest}</span></>
                                  )}
                                </td>
                                <td className={s.mono}>{num(r.variable_gini ?? null, 3)}</td>
                                <td className={`${s.mono} ${r.gini_gain !== null && r.gini_gain > 0 ? s.good : ''}`}>{num(r.gini_gain, 4, true)}</td>
                                <td>
                                  <span className={`${s.outcome} ${failed ? s.outcomeBad : r.status === 'pending' ? '' : s.outcomeGood}`}>{r.status}</span>
                                  {failed && r.reason && <div className={s.reason}>{r.reason.split('\n')[0]}</div>}
                                  {r.note && <div className={s.reason}>{r.note}</div>}
                                </td>
                                <td>
                                  {r.sent && (
                                    <span className={`${s.sent} ${s[EVAL_CLASS[r.sent] ?? '']}`}>
                                      {r.sent}
                                    </span>
                                  )}
                                </td>
                                <td>
                                  {r.code && (
                                    <button className={s.codeBtn} onClick={() => setOpen((o) => ({ ...o, [id]: !o[id] }))}>
                                      {open[id] ? 'hide' : r.kind === 'request' ? 'SQL' : 'code'}
                                    </button>
                                  )}
                                </td>
                              </>
                            )}
                          </tr>
                          {open[id] && (
                            <tr className={s.codeRow}>
                              <td colSpan={N_COLS - IDEA_COLS}><pre className={s.code}>{r.code}</pre></td>
                            </tr>
                          )}
                        </Fragment>
                      )
                })}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </div>
  )
}
