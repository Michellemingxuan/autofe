import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import { api, usePolled } from '../api'
import type { PathCheck, SetupBlock, SetupField, TargetCases, Workspace } from '../types'
import { ConfirmButton } from './ConfirmButton'
import {
  DefaultHint, FieldLabel, FilesList, PathOrUpload, PlainField, Tick, TypedField, inFolder, normalise,
  relTo, usePathCheck,
} from './SetupFields'
import { ShotsBody } from './SetupShots'
import { AddSource, LinkageJob, SourceRow } from './SetupSources'
import s from './SetupPage.module.css'

type Values = Record<string, unknown>
type Readiness = 'ready' | 'attention' | 'edited'

// The order the agent meets things: the task, the model materials, extra data
// and where more may come from - then how features are judged.
const ORDER: SetupBlock['key'][] = ['task', 'model_folder', 'model', 'shots', 'context', 'folder', 'additional',
                                     'scope', 'evaluation', 'cases_folder', 'target_cases']
// The sections, each a heading over its blocks - lettered when there are several.
const SECTIONS: { title: string; keys: SetupBlock['key'][] }[] = [
  { title: 'Overview', keys: ['task'] },
  { title: 'Model materials', keys: ['model_folder', 'model', 'shots', 'context'] },
  { title: 'Additional materials', keys: ['folder', 'additional', 'scope'] },
  { title: 'Evaluation', keys: ['evaluation'] },
  { title: 'Inference', keys: ['cases_folder', 'target_cases'] },
]
const sectionOf = (key: SetupBlock['key']) => SECTIONS.find((g) => g.keys.includes(key))
// Each section's folder comes first: the paths below it are names inside it.
const FOLDERS: SetupBlock['key'][] = ['model_folder', 'folder', 'cases_folder']
const letterOf = (key: SetupBlock['key']) => {
  if (FOLDERS.includes(key)) return null                // the section's folder, ahead of its blocks
  const keys = (sectionOf(key)?.keys ?? []).filter((k) => !FOLDERS.includes(k))
  return keys.length > 1 ? 'abcdefgh'[keys.indexOf(key)] : null
}
const READINESS_LABEL: Record<Readiness, string> = {
  ready: 'Ready', attention: 'Needs attention', edited: 'Edits not applied',
}

const asText = (v: unknown) => (Array.isArray(v) ? v.join(',') : String(v ?? ''))
const isDirty = (f: SetupField, v: unknown) =>
  JSON.stringify(normalise(f, v)) !== JSON.stringify(normalise(f, f.value))

/**
 * Step 1. The data the agent works from, as a spine of blocks in the order
 * the agent meets it. Each block shows a readout of what is loaded now,
 * prefills from the config file, and is applied on its own - saved only if
 * the workspace it describes loads.
 */
export function SetupPage({ workspace, onChange }: { workspace: Workspace | null; onChange: () => void }) {
  const [setup, refreshSetup] = usePolled(api.setup, [], 60000)
  const [cases, refreshCases] = usePolled(api.targetCases, [], 30000)
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
  const extraFolder = asText(values['discovery.additional_data.dir'])
  const casesFolder = asText(values['discovery.additional_data.target_cases.dir'])
  // The model materials' folder: where their files lie, not a setting - the
  // dataset's files are names inside it, and moving it moves them.
  const loadedModel = MODEL_FILES.map((k) => asText(setup?.blocks.flatMap((b) => b.fields)
    .find((f) => f.key === k)?.value))
  const [modelFolder, setModelFolder] = useState('')
  useEffect(() => { setModelFolder(commonFolder(loadedModel)) }, [loadedModel.join('\n')])  // eslint-disable-line
  const moveModelFolder = (next: string) => {
    const name = (p: string) => relTo(modelFolder, p)
    for (const k of MODEL_FILES) {
      const n = name(asText(values[k]))
      if (n && !n.startsWith('/')) set(k, inFolder(next.replace(/\/$/, ''), n))
    }
    setModelFolder(next.replace(/\/$/, ''))
  }
  const applied = () => { refreshSetup(); refreshCases(); onChange() }

  // Every path on screen, checked as it is typed.
  const paths = useMemo(() => {
    const out: string[] = [modelFolder]
    for (const b of setup?.blocks ?? []) for (const f of b.fields) {
      if (['path', 'file'].includes(f.kind)) out.push(asText(values[f.key]))
      if (f.kind === 'files') out.push(...((values[f.key] as string[]) ?? []))
    }
    return out
  }, [setup, values, modelFolder])
  const checks = usePathCheck(paths)

  if (!setup) return <div className={s.page}><p className={s.loading}>Loading the setup…</p></div>
  const blocks = ORDER.map((k) => setup.blocks.find((b) => b.key === k)).filter(Boolean) as SetupBlock[]
  const state = (b: SetupBlock): Readiness =>
    b.fields.some((f) => isDirty(f, values[f.key])) ? 'edited' : ready(b, workspace, values, checks, cases)
  const readyCount = blocks.filter((b) => state(b) === 'ready').length

  return (
    <div className={s.page}>
      <header className={s.header}>
        <div className={s.headText}>
          <h1 className={s.title}>Setup</h1>
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
          {blocks.map((b, i) => (
            <Fragment key={b.key}>
            {startsSection(blocks, i) && <li className={s.groupHead}>{sectionOf(b.key)?.title}</li>}
            <li ref={(el) => { sections.current[b.key] = el }}
                className={`${s.stop} ${s.inGroup} ${flash === b.key ? s.flash : ''}`}>
              <span className={`${s.node} ${s['node_' + state(b)]}`} aria-hidden />
              <Block block={b} values={values} state={state(b)} onApplied={applied}>
                {b.key === 'model_folder' && (
                  <div className={s.fieldStack}>
                    <div className={s.field}>
                      <span className={s.fieldLabel}>Model materials folder</span>
                      <div className={s.inputRow}>
                        <input className={s.mono} value={modelFolder} placeholder="/path/to/my_use_case"
                               onChange={(e) => moveModelFolder(e.target.value)} />
                        <Tick path={modelFolder} checks={checks} />
                      </div>
                    </div>
                  </div>
                )}
                {b.key === 'model' && <ModelFields block={b} values={values} set={set} checks={checks} folder={modelFolder} />}
                {b.key === 'task' && <Fields block={b} values={values} set={set} checks={checks} />}
                {b.key === 'context' && (
                  <>
                    <Fields block={b} values={values} set={set} checks={checks} folder={modelFolder} />
                    <ThemePoolPart />
                  </>
                )}
                {b.key === 'shots' && (
                  <ShotsBody block={b} values={values} set={set} checks={checks}
                             workspace={workspace} onApplied={applied} folder={modelFolder} />
                )}
                {b.key === 'evaluation' && <EvaluationFields block={b} values={values} set={set} />}
                {b.key === 'folder' && <Fields block={b} values={values} set={set} checks={checks} />}
                {b.key === 'additional' && (
                  <>
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
                      <AddSource checks={checks} folder={extraFolder} onAdded={onChange} />
                    </section>
                    {jobId && <LinkageJob jobId={jobId} onClose={() => { setJobId(null); onChange() }} />}
                  </>
                )}
                {b.key === 'scope' && <ScopeFields block={b} values={values} set={set} checks={checks} folder={extraFolder} />}
                {b.key === 'cases_folder' && <Fields block={b} values={values} set={set} checks={checks} />}
                {b.key === 'target_cases' && (
                  <TargetCasesFields block={b} values={values} set={set} checks={checks} cases={cases}
                                     folder={casesFolder} workspace={workspace} />
                )}
              </Block>
            </li>
            </Fragment>
          ))}
        </ol>
      </div>

      <aside className={s.overview} aria-label="What the agent has now">
        <div className={s.ovHead}>
          <span className={s.ovTitle}>What the agent has now</span>
          <div className={s.ovProgress} aria-label={`${readyCount} of ${blocks.length} ready`}>
            {blocks.map((b) => (
              <i key={b.key} className={s['ovSeg_' + state(b)]} title={`${titleOf(b)}: ${READINESS_LABEL[state(b)]}`} />
            ))}
          </div>
        </div>
        <SetupMap blocks={blocks} state={state} jump={jump}
                  glance={(b) => <Glance block={b} workspace={workspace} values={values} checks={checks} cases={cases} />} />
      </aside>
      </div>
    </div>
  )
}

// A section's heading goes before the first of its blocks the page has.
const startsSection = (blocks: SetupBlock[], i: number) =>
  i === 0 || sectionOf(blocks[i].key) !== sectionOf(blocks[i - 1].key)
const titleOf = (b: SetupBlock) => (letterOf(b.key) ? `${letterOf(b.key)}. ${b.title}` : b.title)

/** Whether a block has what the agent needs, judged from what is loaded. */
function ready(b: SetupBlock, w: Workspace | null, values: Values, checks: PathCheck,
               cases: TargetCases | null): Readiness {
  if (!w) return 'attention'
  const exists = (k: string) => {
    const p = asText(values[k])
    return !p || checks[p]?.exists !== false
  }
  switch (b.key) {
    case 'model':
      return b.fields.filter((f) => f.kind === 'path').every((f) => exists(f.key)) ? 'ready' : 'attention'
    case 'task':
      return asText(values['discovery.task_description']).trim() ? 'ready' : 'attention'
    case 'context':
      return exists('discovery.task_context_path') && exists('discovery.column_descriptions_path')
        ? 'ready' : 'attention'
    case 'shots':
      return w.shots.some((c) => c.found > 0 && c.kind !== 'error') ? 'ready' : 'attention'
    case 'folder':
      return asText(values['discovery.additional_data.dir']) && exists('discovery.additional_data.dir')
        ? 'ready' : 'attention'
    case 'cases_folder':
      return asText(values['discovery.additional_data.target_cases.dir'])
        && exists('discovery.additional_data.target_cases.dir') ? 'ready' : 'attention'
    case 'additional':
      return w.sources.some((x) => x.state === 'needs_linkage') ? 'attention' : 'ready'
    case 'scope':
      return w.scope_files.length ? 'ready' : 'attention'
    case 'target_cases':
      return cases && !cases.error && !cases.missing_base?.length
        && Object.values(cases.sources).every((x) => !x.needed_by || x.state === 'provided') ? 'ready' : 'attention'
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
          <h2 className={s.blockTitle}>{titleOf(block)}</h2>
          <span className={`${s.state} ${s['state_' + state]}`}>{READINESS_LABEL[state]}</span>
          {changed > 0 && <span className={s.changedNote}>{changed} setting{changed > 1 ? 's' : ''} differ from the config file</span>}
        </div>
        {block.help && <p className={s.blockHelp}>{block.help}</p>}
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

// ------------------------------------------------------------------ at a glance


/** The setup as the agent uses it, drawn: the task and the materials feed the agent,
 *  its features go to evaluation, and the evaluated models score the target cases.
 *  Every box is a block - its state, one glance at it, a click to it. */
function SetupMap({ blocks, state, jump, glance }: {
  blocks: SetupBlock[]; state: (b: SetupBlock) => Readiness; jump: (k: string) => void
  glance: (b: SetupBlock) => React.ReactNode
}) {
  const by = (k: SetupBlock['key']) => blocks.find((b) => b.key === k)
  const worst = (keys: SetupBlock['key'][]): Readiness => {
    const st = keys.map(by).filter(Boolean).map((b) => state(b!))
    return st.includes('attention') ? 'attention' : st.includes('edited') ? 'edited' : 'ready'
  }
  const Dot = ({ st }: { st: Readiness }) => <span className={`${s.ovDot} ${s['ovDot_' + st]}`} aria-hidden />
  const Item = ({ k, name }: { k: SetupBlock['key']; name?: string }) => {
    const b = by(k)
    if (!b) return null
    return (
      <button className={s.mItem} onClick={() => jump(k)} title={`${b.title}: ${READINESS_LABEL[state(b)]}`}>
        <span className={s.mItemHead}><Dot st={state(b)} />{name ?? b.title}</span>
        <span className={s.mGlance}>{glance(b)}</span>
      </button>
    )
  }
  const Group = ({ title, folder, keys, children }: {
    title: string; folder?: SetupBlock['key']; keys: SetupBlock['key'][]; children: React.ReactNode
  }) => (
    <div className={s.mGroup}>
      <button className={s.mGroupHead} onClick={() => jump(folder ?? keys[0])}>
        <Dot st={worst(folder ? [folder, ...keys] : keys)} />{title}</button>
      {children}
    </div>
  )
  // A box of one block: its title, state and glance, without a second heading.
  const Node = ({ title, k, folder }: { title: string; k: SetupBlock['key']; folder?: SetupBlock['key'] }) => {
    const b = by(k)
    if (!b) return null
    return (
      <button className={`${s.mGroup} ${s.mNode}`} onClick={() => jump(k)}
              title={`${b.title}: ${READINESS_LABEL[state(b)]}`}>
        <span className={s.mGroupHead}><Dot st={worst(folder ? [folder, k] : [k])} />{title}</span>
        <span className={`${s.mGlance} ${s.mNodeBody}`}>{glance(b)}</span>
      </button>
    )
  }
  const Arrow = ({ label }: { label?: string }) => (
    <div className={s.mArrow}>{label && <span>{label}</span>}</div>
  )
  return (
    <div className={s.map}>
      <Node title="Task" k="task" />
      <Arrow label="sets the direction" />
      <div className={s.mPair}>
        <Group title="Model materials" folder="model_folder" keys={['model', 'shots', 'context']}>
          <Item k="model" name="Dataset" /><Item k="shots" /><Item k="context" />
        </Group>
        <Group title="Additional" folder="folder" keys={['additional', 'scope']}>
          <Item k="additional" name="Datasets" /><Item k="scope" />
        </Group>
      </div>
      <svg className={s.mJoin} viewBox="0 0 100 24" preserveAspectRatio="none" aria-hidden>
        <path d="M25 0 V10 H50 M75 0 V10 H50 M50 10 V24" />
      </svg>
      <div className={s.mHub}>
        <b>Modeling agent</b>
        <span>proposes ideas, builds and screens features</span>
      </div>
      <Arrow label="verified features" />
      <Node title="Evaluation" k="evaluation" />
      <Arrow label="evaluated models" />
      <Node title="Inference · target cases" k="target_cases" folder="cases_folder" />
    </div>
  )
}

/** A block at a glance: one picture of what is loaded - the details are in the block. */
function Glance({ block, workspace: w, values, checks, cases }: {
  block: SetupBlock; workspace: Workspace | null; values: Values; checks: PathCheck; cases: TargetCases | null
}) {
  if (!w) return null
  const v = (k: string) => values[k]
  const has = (k: string) => { const p = asText(v(k)); return !!p && checks[p]?.exists !== false }
  const Stat = ({ n, label, warn }: { n: React.ReactNode; label: string; warn?: boolean }) => (
    <span className={`${s.gStat} ${warn ? s.gWarn : ''}`}><b>{n}</b>{label}</span>)
  const Pill = ({ ok, label, title }: { ok: boolean | 'warn'; label: string; title?: string }) => (
    <span className={`${s.gPill} ${ok === true ? s.gOk : ok === 'warn' ? s.gAmber : s.gOff}`} title={title}>
      {ok === true ? '✓' : ok === 'warn' ? '!' : '✕'} {label}</span>)
  switch (block.key) {
    case 'task':
      return asText(v('discovery.task_description')).trim()
        ? <span className={s.gText}>{asText(v('discovery.task_description')).trim()}</span>
        : <Pill ok={false} label="not described" />
    case 'model':
      return <div className={s.gRow}>
        <Pill ok={['data.paths.train', 'data.paths.valid', 'data.paths.test'].every(has)}
              label="model building splits" title="train · valid · test" />
        <Pill ok={['discovery.screen_paths.train', 'discovery.screen_paths.valid'].every(has)}
              label="quick screening" title="the screen's fit and score samples" />
      </div>
    case 'shots':
      return <div className={s.gRow}>
        <Pill ok={w.shots.some((c) => c.kind === 'clustering' && c.found > 0)} label="clustering" />
      </div>
    case 'context':
      return <div className={s.gRow}>
        <Pill ok={has('discovery.task_context_path')} label="description" />
        <Pill ok={has('discovery.column_descriptions_path')} label="columns" />
      </div>
    case 'folder': {
      const dir = asText(v('discovery.additional_data.dir'))
      return dir ? <code className={s.gCode} title={dir}>{dir.split('/').slice(-2).join('/')}</code>
                 : <Pill ok={false} label="not set" />
    }
    case 'additional':
      return w.sources.length
        ? <div className={s.gRow}>{w.sources.map((x) => (
            <Pill key={x.name} ok={x.state === 'linked' ? true : x.state === 'needs_linkage' ? 'warn' : false}
                  label={x.name} title={x.state.replace('_', ' ')} />))}</div>
        : <Pill ok={false} label="no sources" />
    case 'scope': {
      const scopes = Object.entries(w.scopes ?? {})
      if (!scopes.length) return <span className={s.gText}>none - only requests beyond scope</span>
      return <div className={s.gCol}>{scopes.map(([name, c]) => {
        const used = (c.in_model ?? 0) + (c.in_model_unused ?? 0), raw = c.unused_raw ?? 0
        return (
          <div key={name} className={s.gScope} title={`${used} in the model · ${raw} unused`}>
            <span className={s.gScopeName}>{name}</span>
            <span className={s.gBar}><i style={{ flex: used || 0.0001 }} /><em style={{ flex: raw || 0.0001 }} /></span>
            <span className={s.gScopeN}>{used + raw}</span>
          </div>
        )
      })}</div>
    }
    case 'evaluation':
      return null
    case 'target_cases': {
      if (!cases) return null
      if (cases.error) return <Pill ok={false} label="not set up" title={cases.error} />
      return <div className={s.gRow}><Stat n={cases.cases} label="cases" /><Stat n={cases.rows} label="transactions" /></div>
    }
  }
  return null
}


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
      {!pool.ready && <p className={s.empty}>Describe the task in the Overview first.</p>}
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

/** The Target cases block: the cases' settings, then the cases' own data for each
 *  additional dataset - the same datasets, in the same order, as the Additional
 *  datasets block; each by default under its own file name in the cases' folder. */
function TargetCasesFields({ block, values, set, checks, cases, folder, workspace }: {
  block: SetupBlock; values: Values; set: (k: string, v: unknown) => void; checks: PathCheck
  cases: TargetCases | null; folder: string; workspace: Workspace | null
}) {
  const histories = block.fields.filter((f) => f.section === 'Histories')
  const linkage = Object.fromEntries((workspace?.sources ?? []).map((x) => [x.name, x.state]))
  return (
    <>
      <Fields block={{ ...block, fields: block.fields.filter((f) => f.section !== 'Histories') }}
              values={values} set={set} checks={checks} folder={folder} />
      {cases?.error && <p className={s.error}>{cases.error}</p>}
      {cases && !cases.error && (
        <div className={s.casesLine}>
          <span><b>{cases.cases?.toLocaleString()}</b> cases</span>
          <span><b>{cases.rows?.toLocaleString()}</b> transactions{cases.per_case ? `, ${cases.per_case.min}–${cases.per_case.max} a case` : ''}</span>
          {cases.id_example && <IdCheck built={cases.id_example} model={cases.model_id_example} />}
          {!!cases.missing_base?.length && (
            <span className={s.warnText}>lacks {cases.missing_base.length} base column(s): {cases.missing_base.slice(0, 5).join(', ')}</span>
          )}
        </div>
      )}
      <section className={s.part}>
        <h3 className={s.partTitle}>The cases' additional datasets</h3>
        <p className={s.empty}>One per additional dataset, in its format - by default its own file name in the
          folder above. A dataset the verified features read must be there.</p>
        <div className={s.histories}>
          {histories.map((f) => {
            const st = cases?.sources[f.label]
            const value = asText(values[f.key])
            const needed = (st?.needed_by ?? 0) > 0
            const there = !!value && checks[value]?.exists !== false
            const state = linkage[f.label]
            return (
              <div key={f.key} className={s.history}>
                <span className={s.historyName}>
                  <span><code>{f.label}</code>
                    {state && <span className={`${s.linkTag} ${s['link_' + state]}`}>{state.replace('_', ' ')}</span>}</span>
                  <span className={s.empty}>{needed ? `read by ${st!.needed_by} feature${st!.needed_by === 1 ? '' : 's'}` : 'not read yet'}</span>
                </span>
                <input className={s.mono} value={relTo(folder, value)} title={value}
                       placeholder={f.source_file ?? `the cases' ${f.label}`}
                       onChange={(e) => set(f.key, inFolder(folder, e.target.value))} />
                <span className={`${s.historyState} ${there ? s.hsOk : needed ? s.hsBad : s.hsOff}`}>
                  {there ? '✓ found' : value ? (needed ? 'not found' : 'none') : needed ? 'needed' : 'optional'}
                </span>
              </div>
            )
          })}
          {!histories.length && <p className={s.empty}>No additional datasets yet.</p>}
        </div>
      </section>
    </>
  )
}

/** An id's shape: its parts, and what each part is - digits (a date's eight
 *  exactly), letters or a mix - with the separators between them. The last part,
 *  the indicator that tells a day's transactions apart, may differ in form
 *  between the cases and the model data, so only its presence counts. */
function idShape(id: string): string {
  const parts = id.split(/([^A-Za-z0-9]+)/)
  return parts.map((p, i) => {
    if (i === parts.length - 1 && i > 0) return 'indicator'
    if (i % 2) return p
    if (/^\d+$/.test(p)) return p.length === 8 ? 'date' : 'digits'
    return /^[A-Za-z]+$/.test(p) ? 'letters' : 'mixed'
  }).join('')
}

/** The id built for a case beside an id of the model data: they must be shaped
 *  alike, or the cases' rows meet no feature and no history. */
function IdCheck({ built, model }: { built: string; model?: string | null }) {
  const same = !model || idShape(built) === idShape(model)
  return (
    <span className={same ? undefined : s.warnText}
          title="The cases have no model id: it is built from their columns by the id template. It must be shaped like the model data's ids.">
      In cases, <code>{built}</code>; in model dataset, <code>{model ?? '-'}</code>
      {model && (same ? <span className={s.idOk}> ✓ same shape</span>
        : <> - shapes differ: edit <code>id_template</code> in the config</>)}
    </span>
  )
}

/** One section per scope, under its keyword - the same three settings each. */
function ScopeFields({ block, values, set, checks, folder }: {
  block: SetupBlock; values: Values; set: (k: string, v: unknown) => void; checks: PathCheck; folder: string
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
                  values={values} set={set} checks={checks} folder={folder} />
        </section>
      ))}
    </>
  )
}

function Fields({ block, values, set, checks, folder }: {
  block: SetupBlock; values: Values; set: (k: string, v: unknown) => void; checks: PathCheck; folder?: string
}) {
  return (
    <div className={s.fieldStack}>
      {block.fields.map((f) => {
        const dirty = isDirty(f, values[f.key])
        if (f.kind === 'files') {
          return <FilesList key={f.key} field={f} value={(values[f.key] as string[]) ?? []}
                            onChange={(v) => set(f.key, v)} checks={checks} dirty={dirty} folder={folder} />
        }
        if (f.kind === 'file') {
          return <PathOrUpload key={f.key} field={f} value={asText(values[f.key])} folder={folder}
                               onChange={(v) => set(f.key, v)} checks={checks} dirty={dirty} />
        }
        return <PlainField key={f.key} field={f} value={asText(values[f.key])} folder={folder}
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
function ModelFields({ block, values, set, checks, folder }: {
  block: SetupBlock; values: Values; set: (k: string, v: unknown) => void; checks: PathCheck; folder: string
}) {
  const fileFields = block.fields.filter((f) => f.kind === 'path')
  const otherFields = block.fields.filter((f) => f.kind !== 'path')
  const nameOf = (path: string) => relTo(folder, path)
  const join = (name: string) => inFolder(folder, name)

  return (
    <div className={s.fieldStack}>
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

// The model dataset's files: the model materials' folder is where they lie.
const MODEL_FILES = ['data.paths.train', 'data.paths.valid', 'data.paths.test',
                     'discovery.screen_paths.train', 'discovery.screen_paths.valid']

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
