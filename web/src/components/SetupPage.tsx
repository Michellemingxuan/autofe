import { useEffect, useMemo, useRef, useState } from 'react'
import { api, usePolled } from '../api'
import type { PathCheck, SetupBlock, SetupField, Workspace } from '../types'
import { ConfirmButton } from './ConfirmButton'
import {
  DefaultHint, FieldLabel, FilesList, PathOrUpload, PlainField, Tick, TypedField, normalise,
  usePathCheck,
} from './SetupFields'
import { ShotOverview, ShotsBody, removeShot } from './SetupShots'
import { AddSource, LinkageJob, SourceRow } from './SetupSources'
import s from './SetupPage.module.css'

type Values = Record<string, unknown>
type Readiness = 'ready' | 'attention' | 'edited'

// The order data reaches the agent: the model rows, what they mean, examples,
// extra data and where it may come from - then how features are judged.
const ORDER: SetupBlock['key'][] = ['model', 'context', 'shots', 'additional', 'scope', 'evaluation']
const READINESS_LABEL: Record<Readiness, string> = {
  ready: 'Ready', attention: 'Needs attention', edited: 'Edits not applied',
}

const asText = (v: unknown) => (Array.isArray(v) ? v.join(',') : String(v ?? ''))
const isDirty = (f: SetupField, v: unknown) =>
  JSON.stringify(normalise(f, v)) !== JSON.stringify(normalise(f, f.value))

/**
 * Step 1. The data the agent works from, as a spine of blocks in the order
 * the data reaches it. Each block shows a readout of what is loaded now,
 * prefills from the config file, and is applied on its own - saved only if
 * the workspace it describes loads.
 */
export function SetupPage({ workspace, onChange }: { workspace: Workspace | null; onChange: () => void }) {
  const [setup, refreshSetup] = usePolled(api.setup, [], 60000)
  const [values, setValues] = useState<Values>({})
  const [jobId, setJobId] = useState<string | null>(null)
  const sections = useRef<Record<string, HTMLElement | null>>({})
  const [flash, setFlash] = useState<string | null>(null)
  // Jump, then mark where you landed - an instant jump is reliable everywhere,
  // and the brief ring shows which block the map took you to.
  const jump = (key: string) => {
    sections.current[key]?.scrollIntoView({ block: 'start' })
    setFlash(key)
    setTimeout(() => setFlash((f) => (f === key ? null : f)), 1400)
  }

  useEffect(() => {
    if (setup) setValues(Object.fromEntries(setup.blocks.flatMap((b) => b.fields.map((f) => [f.key, f.value]))))
  }, [setup])
  useEffect(() => { if (workspace?.active_linkage) setJobId(workspace.active_linkage) },
            [workspace?.active_linkage])

  const set = (key: string, v: unknown) => setValues((old) => ({ ...old, [key]: v }))
  const applied = () => { refreshSetup(); onChange() }

  // Every path on screen, checked as it is typed.
  const paths = useMemo(() => {
    const out: string[] = []
    for (const b of setup?.blocks ?? []) for (const f of b.fields) {
      if (['path', 'file'].includes(f.kind)) out.push(asText(values[f.key]))
      if (f.kind === 'files') out.push(...((values[f.key] as string[]) ?? []))
    }
    return out
  }, [setup, values])
  const checks = usePathCheck(paths)

  if (!setup) return <div className={s.page}><p className={s.loading}>Loading the setup…</p></div>
  const blocks = ORDER.map((k) => setup.blocks.find((b) => b.key === k)).filter(Boolean) as SetupBlock[]
  const state = (b: SetupBlock): Readiness =>
    b.fields.some((f) => isDirty(f, values[f.key])) ? 'edited' : ready(b, workspace, values, checks)
  const readyCount = blocks.filter((b) => state(b) === 'ready').length

  return (
    <div className={s.page}>
      <header className={s.header}>
        <div className={s.headText}>
          <h1 className={s.title}>Setup</h1>
          <p className={s.subtitle}>The data the agent works from, and how its features will be judged.</p>
        </div>
        <span className={s.mapCount}><b>{readyCount}</b> of {blocks.length} ready</span>
        <ConfirmButton label="Reset to config file" confirm="Drop every saved change?"
                       disabled={!setup.blocks.some((b) => b.fields.some((f) => f.changed))}
                       title={`Defaults come from ${setup.config_file}`}
                       onConfirm={async () => { await api.resetSetup(); applied() }} />
      </header>
      {setup.error && <div className={s.banner}>{setup.error}</div>}

      <div className={s.split}>
      <div className={s.scroller}>
        <ol className={s.spine}>
          {blocks.map((b) => (
            <li key={b.key} ref={(el) => { sections.current[b.key] = el }}
                className={`${s.stop} ${flash === b.key ? s.flash : ''}`}>
              <span className={`${s.node} ${s['node_' + state(b)]}`} aria-hidden />
              <Block block={b} values={values} state={state(b)} onApplied={applied}>
                {b.key === 'model' && <ModelFields block={b} values={values} set={set} checks={checks} />}
                {b.key === 'context' && (
                  <>
                    <Fields block={b} values={values} set={set} checks={checks} />
                    <ThemePoolPart />
                  </>
                )}
                {b.key === 'shots' && (
                  <ShotsBody block={b} values={values} set={set} checks={checks}
                             workspace={workspace} onApplied={applied} />
                )}
                {b.key === 'evaluation' && <EvaluationFields block={b} values={values} set={set} />}
                {b.key === 'additional' && (
                  <>
                    <Fields block={b} values={values} set={set} checks={checks} />
                    <section className={s.part}>
                      <h3 className={s.partTitle}>Sources</h3>
                      <div className={s.sources}>
                        {workspace?.sources.length === 0 && (
                          <p className={s.empty}>No sources yet. Drop data and its sample JSON in the folder, or add one below.</p>
                        )}
                        {workspace?.sources.map((src) => (
                          <SourceRow key={src.name} source={src}
                                     busy={!!(workspace.active_run || workspace.active_linkage)}
                                     onPropose={async () => {
                                       const { run_id } = await api.proposeLinkage(src.name)
                                       setJobId(run_id)
                                       onChange()
                                     }}
                                     onRemove={async () => { await api.removeSource(src.name); onChange() }} />
                        ))}
                      </div>
                      <AddSource checks={checks} onAdded={onChange} />
                    </section>
                    {jobId && <LinkageJob jobId={jobId} onClose={() => { setJobId(null); onChange() }} />}
                  </>
                )}
                {b.key === 'scope' && <ScopeFields block={b} values={values} set={set} checks={checks} />}
              </Block>
            </li>
          ))}
        </ol>
      </div>

      <aside className={s.overview} aria-label="What the agent has now">
        <div className={s.ovHead}>
          <span className={s.ovTitle}>What the agent has now</span>
          <span className={s.ovSub}>Live from the loaded setup. Click a section to edit it.</span>
        </div>
        {blocks.map((b) => (
          <section key={b.key} className={s.ovSection}>
            <button className={s.ovSectionHead} onClick={() => jump(b.key)} title={READINESS_LABEL[state(b)]}>
              <span className={s.ovName}>{b.title}</span>
              <span className={`${s.ovState} ${s['state_' + state(b)]}`}>{READINESS_LABEL[state(b)]}</span>
            </button>
            {b.key === 'shots'
              ? <ShotOverview cats={workspace?.shots ?? []} target={workspace?.target ?? 'target'}
                              onRemove={async (c) => { await removeShot(c, b); applied() }} />
              : <Readout block={b} workspace={workspace} values={values} checks={checks} />}
          </section>
        ))}
      </aside>
      </div>
    </div>
  )
}

/** Whether a block has what the agent needs, judged from what is loaded. */
function ready(b: SetupBlock, w: Workspace | null, values: Values, checks: PathCheck): Readiness {
  if (!w) return 'attention'
  const exists = (k: string) => {
    const p = asText(values[k])
    return !p || checks[p]?.exists !== false
  }
  switch (b.key) {
    case 'model':
      return b.fields.filter((f) => f.kind === 'path').every((f) => exists(f.key)) ? 'ready' : 'attention'
    case 'context':
      return asText(values['discovery.task_description']).trim() && exists('discovery.task_context_path')
        ? 'ready' : 'attention'
    case 'shots':
      return w.shots.some((c) => c.found > 0 && c.kind !== 'error') ? 'ready' : 'attention'
    case 'additional':
      return w.sources.some((x) => x.state === 'needs_linkage') ? 'attention' : 'ready'
    case 'scope':
      return w.scope_files.length ? 'ready' : 'attention'
    default:
      return 'ready'
  }
}

/** One stop on the spine: what is loaded now, its fields, and its own Apply. */
function Block({ block, values, state, onApplied, children }: {
  block: SetupBlock; values: Values; state: Readiness
  onApplied: () => void; children: React.ReactNode
}) {
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null)
  const edited = block.fields.filter((f) => isDirty(f, values[f.key]))
  const changed = block.fields.filter((f) => f.changed).length

  const apply = async () => {
    setSaving(true)
    setMessage(null)
    try {
      await api.applySetup(Object.fromEntries(edited.map((f) => [f.key, values[f.key]])))
      setMessage({ ok: true, text: 'Saved. The workspace loads with these settings.' })
      onApplied()
    } catch (e: any) {
      setMessage({ ok: false, text: e.message })
    } finally {
      setSaving(false)
    }
  }
  return (
    <article className={`${s.block} ${s['block_' + block.key]}`}>
      <header className={s.blockHead}>
        <div className={s.blockTitleRow}>
          <h2 className={s.blockTitle}>{block.title}</h2>
          <span className={`${s.state} ${s['state_' + state]}`}>{READINESS_LABEL[state]}</span>
          {changed > 0 && <span className={s.changedNote}>{changed} setting{changed > 1 ? 's' : ''} differ from the config file</span>}
        </div>
        <p className={s.blockHelp}>{block.help}</p>
      </header>
      <div className={s.blockBody}>{children}</div>
      {block.fields.length > 0 && (
        <footer className={`${s.blockFoot} ${edited.length ? s.footActive : ''}`}>
          {message
            ? <span className={message.ok ? s.okText : s.errorText}>{message.text}</span>
            : <span className={s.footNote}>{edited.length
                ? `${edited.length} edit${edited.length > 1 ? 's' : ''}: ${edited.map((f) => f.label).join(', ')}`
                : 'No edits'}</span>}
          <button className={s.primary} disabled={saving || !edited.length} onClick={apply}>
            {saving ? 'Checking…' : 'Apply'}
          </button>
        </footer>
      )}
    </article>
  )
}

// ------------------------------------------------------------------ readouts

/** The block's state in one glance, drawn from what is loaded now. */
function Readout({ block, workspace: w, values, checks }: {
  block: SetupBlock; workspace: Workspace | null; values: Values; checks: PathCheck
}) {
  if (!w) return null
  switch (block.key) {
    case 'model': {
      const files = block.fields.filter((f) => f.kind === 'path').map((f) => {
        const c = checks[asText(values[f.key])]
        return { label: f.label.replace(/ \(.*\)$/, ''), bytes: c?.bytes ?? 0, ok: c?.exists !== false }
      })
      const total = files.reduce((n, f) => n + f.bytes, 0) || 1
      return (
        <div className={s.rdModel}>
          <div className={s.sizeBar}>
            {files.map((f, i) => (
              <span key={f.label} className={`${s.sizePart} ${s['tone' + i]} ${f.ok ? '' : s.sizeMissing}`}
                    style={{ flex: Math.max(f.bytes / total, 0.04) }} title={`${f.label}: ${fmtBytes(f.bytes)}`} />
            ))}
          </div>
          <ul className={s.legend}>
            {files.map((f, i) => (
              <li key={f.label}><i className={s['tone' + i]} />{f.label}
                <span className={f.ok ? undefined : s.warnText}>{f.ok ? fmtBytes(f.bytes) : 'missing'}</span></li>
            ))}
          </ul>
          <dl className={s.facts}>
            <div><dt>Base features</dt><dd>{w.base_features}</dd></div>
            <div><dt>Target</dt><dd><code>{w.target}</code></dd></div>
            <div><dt>Screen fit</dt><dd>{w.screen_rows.fit.toLocaleString()} rows</dd></div>
            <div><dt>Screen score</dt><dd>{w.screen_rows.scored.toLocaleString()} rows</dd></div>
          </dl>
        </div>
      )
    }
    case 'context': {
      const task = asText(values['discovery.task_description']).trim()
      return task
        ? <blockquote className={s.quote}>{task.length > 220 ? task.slice(0, 220) + '…' : task}</blockquote>
        : <p className={s.rdLine}>No task description yet - the agent needs to know what to improve.</p>
    }
    case 'shots': {
      const rows = w.shots.reduce((n, c) => n + c.found, 0)
      return (
        <p className={s.rdLine}>
          {w.shots.length
            ? <><b>{w.shots.length}</b> categor{w.shots.length === 1 ? 'y' : 'ies'}, <b>{rows}</b> example rows:{' '}
                {w.shots.map((c) => c.name).join(', ')}.</>
            : 'No examples yet. The agent can still sample rows, but curated shots steer it better.'}
        </p>
      )
    }
    case 'additional':
      return w.sources.length ? (
        <div className={s.chips}>
          {w.sources.map((x) => (
            <span key={x.name} className={`${s.srcChip} ${s['src_' + x.state]}`}>
              {x.name}<i>{x.state === 'linked' ? 'linked' : x.state === 'needs_linkage' ? 'needs linkage' : 'schema only'}</i>
            </span>
          ))}
        </div>
      ) : <p className={s.rdLine}>No sources found in the folder.</p>
    case 'scope': {
      const scopes = Object.entries(w.scopes ?? {})
      if (!scopes.length) return <p className={s.rdLine}>No scope is set up: only requests beyond scope are possible.</p>
      return (
        <>
          {scopes.map(([name, counts]) => {
            const used = counts.in_model ?? 0, quiet = counts.in_model_unused ?? 0, raw = counts.unused_raw ?? 0
            if (!(used + quiet + raw)) return <p key={name} className={s.rdLine}><b>{name}</b>: no variable lists found.</p>
            return (
              <div key={name} className={s.rdScope}>
                <div className={s.scopeBar}>
                  <span style={{ flex: used }} className={s.scopeUsed} />
                  {quiet > 0 && <span style={{ flex: quiet }} className={s.scopeQuiet} />}
                  <span style={{ flex: raw }} className={s.scopeRaw} />
                </div>
                <p className={s.rdLine}>
                  <b>{name}</b>: <b>{used.toLocaleString()}</b> variables already in the model
                  {quiet > 0 && <>, <b>{quiet.toLocaleString()}</b> in it without importance</>},{' '}
                  <b>{raw.toLocaleString()}</b> raw and unused - the room the agent has.
                </p>
              </div>
            )
          })}
          {w.scope_notes.length > 0 && (
            <p className={s.rdLine}>Your notes: {w.scope_notes.map((n) => n.name).join(', ')}.</p>
          )}
        </>
      )
    }
    case 'evaluation': {
      const v = (k: string) => values[k]
      const pct = (normalise({ kind: 'numbers' } as SetupField, v('analysis.capture_rate_percents')) as number[])
        .map((p) => `${+(p * 100).toFixed(2)}%`)
      return (
        <p className={s.rdSentence}>
          Each feature joins base in an XGBoost model - depth {String(v('model.params.max_depth'))},
          learning rate {String(v('model.params.eta'))}, up to {String(v('model.num_boost_round'))} rounds
          {v('model.tuning.enabled') === true ? ', tuned first' : ''}. Reported on test: Gini gain
          {pct.length ? <> and capture at the top {pct.join(', ')}</> : null}
          {v('analysis.shap.enabled') === true ? ', with SHAP ranks' : ''}. A feature passes with a Gini
          gain of at least {String(v('verdict.min_gini_gain'))}; gates are {String(v('run.gates'))}.
        </p>
      )
    }
  }
}

const fmtBytes = (b: number) =>
  !b ? '–' : b > 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`

// -------------------------------------------------------------------- fields

/** The themes an open exploration draws from - made from the task description. */
function ThemePoolPart() {
  const [pool, refresh] = usePolled(api.themes, [], 3000)
  const [error, setError] = useState<string | null>(null)
  if (!pool) return null
  const make = async () => {
    setError(null)
    try { await api.makeThemes() } catch (e: any) { setError(e.message) }
    refresh()
  }
  return (
    <section className={s.part}>
      <h3 className={s.partTitle}>Themes for open explorations</h3>
      <p className={s.empty}>
        A direction left empty explores: each round the agent draws one of these themes, the least
        explored first. They are made from the task description, context, columns, sources and
        scopes - again whenever those change.
      </p>
      {!pool.ready && <p className={s.empty}>Describe the task above first.</p>}
      {pool.generating && <p className={s.empty}>Making the themes…</p>}
      {pool.stale && !pool.generating && <p className={s.empty}>Made for an earlier task description.</p>}
      {(pool.error || error) && <p className={s.error}>{pool.error || error}</p>}
      {pool.themes.length > 0 && (
        <div className={s.themeChips}>
          {pool.themes.map((t) => (
            <span key={t.theme} className={`${s.themeChip} ${t.runs ? s.themeUsed : ''}`}
                  title={t.runs ? `explored by ${t.runs} run${t.runs === 1 ? '' : 's'}` : 'not explored yet'}>
              {t.theme}{t.runs > 0 && <i>{t.runs}</i>}
            </span>
          ))}
        </div>
      )}
      {pool.ready && (
        <div><button className={s.secondary} disabled={pool.generating} onClick={make}>
          {pool.themes.length ? 'Make the themes again' : 'Make the themes'}</button></div>
      )}
    </section>
  )
}

/** One section per scope, under its keyword - the same three settings each. */
function ScopeFields({ block, values, set, checks }: {
  block: SetupBlock; values: Values; set: (k: string, v: unknown) => void; checks: PathCheck
}) {
  const scopes = [...new Set(block.fields.map((f) => f.section ?? 'Scope'))]
  if (!scopes.length) {
    return <p className={s.empty}>No scope is configured. Add one under discovery.additional_data.scopes
      in the config file, named by its keyword (CAS, say).</p>
  }
  return (
    <>
      {scopes.map((name) => (
        <section key={name} className={s.part}>
          <h3 className={s.partTitle}>{name}</h3>
          <Fields block={{ ...block, fields: block.fields.filter((f) => (f.section ?? 'Scope') === name) }}
                  values={values} set={set} checks={checks} />
        </section>
      ))}
    </>
  )
}

function Fields({ block, values, set, checks }: {
  block: SetupBlock; values: Values; set: (k: string, v: unknown) => void; checks: PathCheck
}) {
  return (
    <div className={s.fieldStack}>
      {block.fields.map((f) => {
        const dirty = isDirty(f, values[f.key])
        if (f.kind === 'files') {
          return <FilesList key={f.key} field={f} value={(values[f.key] as string[]) ?? []}
                            onChange={(v) => set(f.key, v)} checks={checks} dirty={dirty} />
        }
        if (f.kind === 'file') {
          return <PathOrUpload key={f.key} field={f} value={asText(values[f.key])}
                               onChange={(v) => set(f.key, v)} checks={checks} dirty={dirty} />
        }
        return <PlainField key={f.key} field={f} value={asText(values[f.key])}
                           onChange={(v) => set(f.key, v)} checks={checks} dirty={dirty} />
      })}
    </div>
  )
}

/** The evaluation settings, grouped as the backend groups them. */
function EvaluationFields({ block, values, set }: {
  block: SetupBlock; values: Values; set: (k: string, v: unknown) => void
}) {
  const groups = [...new Set(block.fields.map((f) => f.section ?? 'Settings'))]
  return (
    <div className={s.evalGroups}>
      {groups.map((g) => (
        <section key={g} className={s.part}>
          <h3 className={s.partTitle}>{g}</h3>
          <div className={s.evalGrid}>
            {block.fields.filter((f) => (f.section ?? 'Settings') === g).map((f) => (
              <TypedField key={f.key} field={f} value={values[f.key]}
                          onChange={(v) => set(f.key, v)} dirty={isDirty(f, values[f.key])} />
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}

/** The split files share a folder: set it once, then only file names. */
function ModelFields({ block, values, set, checks }: {
  block: SetupBlock; values: Values; set: (k: string, v: unknown) => void; checks: PathCheck
}) {
  const fileFields = block.fields.filter((f) => f.kind === 'path')
  const otherFields = block.fields.filter((f) => f.kind !== 'path')
  const loadedPaths = fileFields.map((f) => asText(f.value))
  const [folder, setFolder] = useState(() => commonFolder(loadedPaths))
  useEffect(() => { setFolder(commonFolder(loadedPaths)) }, [loadedPaths.join('\n')])  // eslint-disable-line

  const nameOf = (path: string) => (folder && path.startsWith(folder + '/') ? path.slice(folder.length + 1) : path)
  const join = (name: string) => (!name || name.startsWith('/') || !folder ? name : `${folder.replace(/\/$/, '')}/${name}`)
  const moveFolder = (next: string) => {
    for (const f of fileFields) set(f.key, next && !nameOf(asText(values[f.key])).startsWith('/')
      ? `${next.replace(/\/$/, '')}/${nameOf(asText(values[f.key]))}` : asText(values[f.key]))
    setFolder(next.replace(/\/$/, ''))
  }

  return (
    <div className={s.fieldStack}>
      <div className={s.field}>
        <span className={s.fieldLabel}>Folder</span>
        <div className={s.inputRow}>
          <input className={s.mono} value={folder} placeholder="data/my_use_case"
                 onChange={(e) => moveFolder(e.target.value)} />
          <Tick path={folder} checks={checks} />
        </div>
        <span className={s.fieldHelp}>The files below are names inside this folder; start a name with / to point elsewhere.</span>
      </div>
      {[
        { title: 'Splits', keys: ['data.paths.train', 'data.paths.valid', 'data.paths.test'] },
        { title: 'Screen sample', keys: ['discovery.screen_paths.train', 'discovery.screen_paths.valid'] },
      ].map((row) => (
        <div key={row.title} className={s.fileRowGroup}>
          <span className={s.rowTitle}>{row.title}</span>
          <div className={s.fileGrid} style={{ gridTemplateColumns: `repeat(${row.keys.length}, minmax(0, 1fr))` }}>
            {fileFields.filter((f) => row.keys.includes(f.key)).map((f) => {
              const full = asText(values[f.key])
              return (
                <div key={f.key} className={s.fileRow}>
                  <FieldLabel field={f} dirty={isDirty(f, values[f.key])} />
                  <div className={s.inputRow}>
                    <input className={s.mono} value={nameOf(full)}
                           onChange={(e) => set(f.key, join(e.target.value))} />
                    <Tick path={full} checks={checks} />
                  </div>
                  <DefaultHint field={f} value={full} />
                </div>
              )
            })}
          </div>
        </div>
      ))}
      <div className={s.evalGrid}>
        {otherFields.map((f) => (
          <PlainField key={f.key} field={f} value={asText(values[f.key])}
                      onChange={(v) => set(f.key, v)} checks={checks} dirty={isDirty(f, values[f.key])} />
        ))}
      </div>
    </div>
  )
}

function commonFolder(paths: string[]): string {
  const dirs = paths.filter(Boolean).map((p) => p.split('/').slice(0, -1))
  if (!dirs.length) return ''
  const first = dirs[0]
  let n = first.length
  for (const d of dirs) {
    let i = 0
    while (i < n && i < d.length && d[i] === first[i]) i++
    n = i
  }
  return first.slice(0, n).join('/')
}
