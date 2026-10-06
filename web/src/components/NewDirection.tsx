import { useEffect, useState } from 'react'
import { api, rerunDraft, type RerunDraft } from '../api'
import { navigate } from '../App'
import type { Level, Params, Workspace } from '../types'
import { SOURCE_STATE } from './Rail'
import s from './NewDirection.module.css'

type Props = {
  workspace: Workspace | null
  busy: boolean
  onStarted: (runId: string) => void
}

const EXAMPLES = [
  'Use payment behaviour relative to spending to find risk the bureau columns miss.',
  'Look for early signs of cash-flow stress in the recent balance trajectory.',
  'Are there CAS authorization variables the model does not use that signal distress?',
]

const LEVELS: { level: Level; title: string; text: string }[] = [
  { level: 'L1', title: 'L1 · model columns or one source', text: 'Combine model columns, or aggregate one extra source joined by its linkage' },
  { level: 'L2', title: 'L2 · source × model', text: 'Combine a source aggregate with model columns' },
  { level: 'L3', title: 'L3 · request data', text: 'SQL for data the model lacks. Alone: a data-request run, nothing screened' },
]

/** Start a direction: the idea in words, and the run's parameters. */
export function NewDirection({ workspace, busy, onStarted }: Props) {
  const [direction, setDirection] = useState('')
  const [params, setParams] = useState<Params | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  // "Edit & re-run" lands here with the earlier run's direction and settings.
  const [rerun] = useState<RerunDraft | null>(() => rerunDraft.take())
  useEffect(() => {
    if (rerun) { setDirection(rerun.direction); setParams(rerun.params) }
  }, [rerun])

  // Defaults arrive with the workspace (the config file's agent section);
  // sources default to the ones whose linkage is confirmed.
  useEffect(() => {
    if (workspace && !params && !rerun) {
      setParams({ ...workspace.defaults,
                  sources: workspace.sources.filter((x) => x.state === 'linked').map((x) => x.name) })
    }
  }, [workspace, params])

  // L3 alone writes SQL; anything else builds features in pandas or spark.
  useEffect(() => {
    if (!params || !workspace) return
    const only = params.levels.length === 1 && params.levels[0] === 'L3'
    if (only && params.engine !== 'sql') setParams({ ...params, engine: 'sql' })
    if (!only && params.engine === 'sql') setParams({ ...params, engine: workspace.defaults.engine })
  }, [params?.levels.join(',')])  // eslint-disable-line

  if (!workspace || !params) return <div className={s.wrap}>Loading…</div>
  const set = <K extends keyof Params>(key: K, value: Params[K]) => setParams({ ...params, [key]: value })
  const toggle = <T,>(list: T[], item: T) =>
    list.includes(item) ? list.filter((x) => x !== item) : [...list, item]

  const start = async () => {
    setError(null)
    setStarting(true)
    try {
      const { run_id } = await api.start(direction.trim(), params, rerun?.replaces)
      onStarted(run_id)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setStarting(false)
    }
  }

  // L3 alone is a data-request run: nothing is screened; the agent proposes SQL.
  const l3Only = params.levels.length === 1 && params.levels[0] === 'L3'
  const noFeatureLevel = params.levels.length === 0
  const needsSource = params.levels.includes('L2') && !params.sources.length

  return (
    <div className={s.wrap}>
      <header className={s.header}>
        <span className={`eyebrow ${s.eyebrow}`}>Step 2 · Discover</span>
        <h1 className={s.title}>{rerun ? 'Re-run a direction' : 'New direction'}</h1>
        {rerun && (
          <p className={s.rerunNote}>
            Starting replaces the earlier run: its features and requests leave the pool, and the
            run is kept on disk under <code>replaced/</code>.
          </p>
        )}
      </header>

      <div className={s.grid}>
        <section className={s.card}>
          <div className={`eyebrow ${s.cardHead}`}>Direction</div>
          <p className={s.lead}>
            Describe how you think the model could be improved. The agent explores the data,
            writes features at the levels you allow, and screens each one against base.
          </p>
          <textarea className={s.text} rows={6} value={direction}
                    placeholder="e.g. customers who pay back less of what they spend are riskier"
                    onChange={(e) => setDirection(e.target.value)} />
          <div className={s.examples}>
            {EXAMPLES.map((x) => (
              <button key={x} className={s.example} onClick={() => setDirection(x)}>{x}</button>
            ))}
          </div>
          <div className={s.context}>
            <span><b>{workspace.base_features}</b> base features</span>
            <span><b>{workspace.screen_rows.fit.toLocaleString()}</b> rows fit ·
              <b> {workspace.screen_rows.scored.toLocaleString()}</b> scored</span>
            <span title={workspace.id_format}>id <code>{workspace.id_column}</code></span>
          </div>
        </section>

        <section className={s.card}>
          <div className={`eyebrow ${s.cardHead}`}>Parameters
            <span className={s.prefill}> · prefilled from the config file</span></div>

          <div className={s.row2}>
            <Field label={l3Only ? 'K requests' : 'K intents'}
                   hint={l3Only ? 'Most data requests the agent may propose' : 'Most features the agent may screen'}>
              <input type="number" min={1} max={50} value={params.K}
                     onChange={(e) => set('K', Math.max(1, Number(e.target.value) || 1))} />
            </Field>
            <Field label="Model">
              <select value={params.model} onChange={(e) => set('model', e.target.value)}>
                {workspace.choices.models.map((m) => <option key={m}>{m}</option>)}
              </select>
            </Field>
          </div>

          {!l3Only && (
          <div className={s.row2}>
            <Field label="Min Gini gain" hint="A verified feature must beat this">
              <input type="number" step={0.001} value={params.min_gini_gain}
                     onChange={(e) => set('min_gini_gain', Number(e.target.value) || 0)} />
            </Field>
            <Field label={`Min capture-rate gain${params.capture_percent ? ` · top ${params.capture_percent * 100}%` : ''}`}
                   hint="…and this, if set; blank = not gated">
              <input type="number" step={0.001} value={params.min_capture_gain ?? ''}
                     onChange={(e) => set('min_capture_gain',
                                          e.target.value === '' ? null : Number(e.target.value))} />
            </Field>
          </div>
          )}

          <Field label="Levels">
            <div className={s.levels}>
              {LEVELS.map((l) => (
                <label key={l.level} className={`${s.level} ${params.levels.includes(l.level) ? s.levelOn : ''}`}>
                  <input type="checkbox" checked={params.levels.includes(l.level)}
                         onChange={() => set('levels', toggle(params.levels, l.level))} />
                  <span className={s.levelText}>
                    <span className={s.levelTitle}>{l.title}</span>
                    <span className={s.levelHint}>{l.text}</span>
                  </span>
                </label>
              ))}
            </div>
          </Field>

          <Field label="Engine" hint={l3Only
            ? 'A data-request run writes BigQuery SQL; its local checks run in ' + workspace.defaults.engine
            : 'What the agent writes its feature code in'}>
            <div className={s.segment}>
              {workspace.choices.engines.map((eng) => {
                const allowed = l3Only ? eng === 'sql' : eng !== 'sql'
                return (
                  <button key={eng} disabled={!allowed} className={params.engine === eng ? s.segOn : ''}
                          title={allowed ? undefined : eng === 'sql' ? 'Only for L3 alone' : 'Not for a data-request run'}
                          onClick={() => set('engine', eng)}>{eng}</button>
                )
              })}
            </div>
          </Field>

          <Field label="Sources">
            <div className={s.sources}>
              {workspace.sources.length === 0 && <span className={s.hint}>No additional data</span>}
              {workspace.sources.map((src) => (
                <label key={src.name} className={s.source} title={SOURCE_STATE[src.state].hint}>
                  <input type="checkbox" checked={params.sources.includes(src.name)}
                         onChange={() => set('sources', toggle(params.sources, src.name))} />
                  <span className={s.sourceName}>{src.name}</span>
                  <span className={`${s.badge} ${s[src.state]}`}>{SOURCE_STATE[src.state].label}</span>
                </label>
              ))}
            </div>
          </Field>

          {params.sources.some((n) => workspace.sources.find((x) => x.name === n)?.state === 'needs_linkage') && (
            <div className={s.note}>
              Some ticked sources have no confirmed linkage yet - confirm them in Setup first, or
              the agent will propose one mid-run for you to approve.
            </div>
          )}
          {noFeatureLevel && <div className={s.warn}>Tick at least one level.</div>}
          {params.levels.length > 1 && (() => {
            const linked = params.sources.some((n) => workspace.sources.find((x) => x.name === n)?.state === 'linked')
            const order = ['L2', 'L1', 'L3'].filter((lv) => params.levels.includes(lv as Level) && (lv !== 'L2' || linked))
            const w = (workspace.defaults.level_weights ?? [0.5, 0.3, 0.2]).slice(0, order.length)
            const total = w.reduce((a, b) => a + b, 0)
            return order.length > 1 ? (
              <div className={s.note}>
                The {params.K} intents are split across levels at random, each drawn with{' '}
                {order.map((lv, i) => `${lv} ${Math.round((w[i] / total) * 100)}%`).join(' · ')}
                {params.levels.includes('L2') && !linked ? ' - L2 is left out: no linked source is ticked' : ''}.
              </div>
            ) : null
          })()}
          {l3Only && (
            <div className={s.note}>
              <b>Data-request run.</b> Nothing is built or screened. The agent reads the CAS scope and
              your notes, then proposes up to {params.K} data pulls - each a rationale, the features it
              would enable, and BigQuery SQL - for you to review and download when it ends.
            </div>
          )}
          {needsSource && <div className={s.warn}>L2 needs at least one source.</div>}
          {error && <div className={s.error}>{error}</div>}

          <div className={s.startRow}>
            <button className={s.start} disabled={!direction.trim() || busy || starting || noFeatureLevel}
                    onClick={start}>
              {busy ? 'A direction is running' : starting ? 'Starting…'
                : rerun ? 'Re-run, replacing the earlier run'
                : l3Only ? 'Start data-request run' : 'Start direction'}
            </button>
            {rerun && (
              // Nothing has changed yet: back to the earlier run, as it was.
              <button className={s.cancel} disabled={starting}
                      onClick={() => navigate('discover', rerun.replaces)}>Cancel</button>
            )}
          </div>
          <p className={s.hint}>
            You confirm each new linkage and every data request; features at L1/L2 run on their own.
          </p>
        </section>
      </div>
    </div>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className={s.field}>
      <div className={s.fieldLabel}>{label}{hint && <span className={s.fieldHint}>{hint}</span>}</div>
      {children}
    </div>
  )
}
