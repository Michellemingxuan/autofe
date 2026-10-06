import { useEffect, useRef, useState } from 'react'
import { api, type JobKind } from '../api'
import type { ApprovalCard, CodeCard, DataRequest, RunView, Step, TraceItem } from '../types'
import { ConfirmButton } from './ConfirmButton'
import { ToolChips } from './ToolChips'
import s from './Trace.module.css'

const KIND_LABEL: Record<Step['kind'], string> = {
  explore: 'Look at the data', ideas: 'Ideas', linkage: 'Linkage', intent: 'Intent', l3: 'Data request',
  challenge: 'Challenge', stage: 'Stage', summary: 'Summary',
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

const signed = (x: number) => `${x >= 0 ? '+' : ''}${x.toFixed(4)}`

/**
 * The agent's work as steps - explore, linkage, each intent, data requests,
 * summary - each with what was said, the scripts it ran and what came back.
 * Finished steps fold to one line; the live step and any step waiting for
 * you stay open. Every script is shown in full when its step is open.
 */
export function Trace({ jobKind, jobId, view, jumpTo = null, onInView, onDeleteIntent }: {
  jobKind: JobKind
  jobId: string
  view: RunView
  jumpTo?: { id: number; n: number } | null
  onInView?: (id: number | null) => void
  onDeleteIntent?: (name: string) => Promise<unknown>
}) {
  const scroller = useRef<HTMLDivElement>(null)
  const pinned = useRef(true)
  const [toggled, setToggled] = useState<Record<number, boolean>>({})

  const lastId = view.steps.length ? view.steps[view.steps.length - 1].id : -1
  const l3 = view.params?.levels?.length === 1 && view.params.levels[0] === 'L3'
  const isOpen = (step: Step) =>
    toggled[step.id] ?? (step.id === lastId || step.status === 'waiting' || step.kind === 'summary')

  // Follow the stream while the reader is at the bottom; stop following when
  // they scroll up, so new events never yank the view away from what they read.
  const onScroll = () => {
    const el = scroller.current
    if (!el) return
    pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
    const top = el.getBoundingClientRect().top
    const steps = [...el.querySelectorAll<HTMLElement>('[data-step]')]
    const visible = steps.find((n) => n.getBoundingClientRect().bottom > top + 40)
    onInView?.(visible ? Number(visible.dataset.step) : null)
  }
  useEffect(() => {
    const el = scroller.current
    if (el && pinned.current) el.scrollTop = el.scrollHeight
  }, [view])

  // A click on the timeline opens that step and brings it to the top.
  useEffect(() => {
    if (!jumpTo) return
    setToggled((t) => ({ ...t, [jumpTo.id]: true }))
    requestAnimationFrame(() => {
      const node = scroller.current?.querySelector(`[data-step="${jumpTo.id}"]`)
      node?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      pinned.current = false
    })
  }, [jumpTo])

  return (
    <section className={s.trace}>
      <div className={s.head}>
        <span className={`eyebrow ${s.headTitle}`}>Agent trace</span>
        <span className={s.headMeta}>
          {view.steps.length} steps · {view.codeCount} scripts
        </span>
      </div>
      <div className={s.scroll} ref={scroller} onScroll={onScroll}>
        {view.steps.length === 0 && <div className={s.empty}>Waiting for the agent…</div>}
        {view.steps.map((step) => (
          <StepCard key={step.id} step={step} job={{ kind: jobKind, id: jobId }} open={isOpen(step)}
                    l3={l3} requests={view.requests}
                    onDeleteIntent={view.status === 'done' ? onDeleteIntent : undefined}
                    onToggle={() => setToggled((t) => ({ ...t, [step.id]: !isOpen(step) }))} />
        ))}
      </div>
    </section>
  )
}

function stepSummary(step: Step): string {
  const count = (k: TraceItem['kind']) => step.items.filter((i) => i.kind === k).length
  const parts = []
  const scripts = count('code')
  if (scripts) parts.push(`${scripts} script${scripts > 1 ? 's' : ''}`)
  const tools = count('tool')
  if (tools) parts.push(`${tools} lookup${tools > 1 ? 's' : ''}`)
  if (step.note) parts.push(step.note)
  return parts.join(' · ')
}

type Job = { kind: JobKind; id: string }

function StepCard({ step, job, open, onToggle, onDeleteIntent, l3 = false, requests = [] }: {
  step: Step; job: Job; open: boolean; onToggle: () => void
  onDeleteIntent?: (name: string) => Promise<unknown>
  l3?: boolean; requests?: DataRequest[]
}) {
  if (l3) return <L3StepCard step={step} job={job} open={open} onToggle={onToggle} requests={requests} />
  const label = step.kind === 'intent' && step.row
    ? `${step.row.intent} · ${step.level}` : step.kind === 'intent' ? `Intent · ${step.level ?? ''}` : KIND_LABEL[step.kind]
  const firstMessage = step.items.find((i) => i.kind === 'message') as { text: string } | undefined

  return (
    <article className={`${s.step} ${s[step.kind]} ${s[step.status]}`} data-step={step.id}>
      <button className={s.stepHead} onClick={onToggle}>
        <span className={`${s.dot} ${s[step.status]}`} />
        <span className={s.kindLabel}>{label}</span>
        <span className={s.stepTitle}>{step.title}</span>
        {step.row?.delta != null && <span className={s.delta}>Δ {signed(step.row.delta)}</span>}
        <span className={`${s.status} ${s[step.status]}`}>{STATUS_LABEL[step.status]}</span>
        <span className={s.chev}>{open ? '▾' : '▸'}</span>
      </button>

      {!open && (
        <div className={s.folded}>
          {firstMessage && <span className={s.foldedText}>{firstMessage.text}</span>}
          <span className={s.foldedMeta}>{stepSummary(step)}</span>
        </div>
      )}

      {open && (
        <div className={s.stepBody}>
          {step.items.map((item) => <Item key={`${item.kind}-${item.seq}`} item={item} job={job} />)}
          {step.row && (
            <div className={`${s.result} ${step.row.verified ? s.resultOk : s.resultNo}`}>
              <div className={s.resultTop}>
                <span className={s.resultName}>{step.row.name}</span>
                <span className={s.resultDelta}>
                  {step.row.delta == null ? 'no score' : `Gini Δ ${signed(step.row.delta)}`}
                  {step.row.capture_gain != null && ` · capture Δ ${signed(step.row.capture_gain)}`}
                </span>
                <span className={s.resultVerdict}>
                  {step.row.deleted ? 'removed from pool' : step.row.verified ? 'verified' : 'not verified'}
                </span>
                {step.row.verified && !step.row.deleted && onDeleteIntent && (
                  <ConfirmButton tone="dark" label="Remove from pool" confirm="Remove this intent?"
                                 onConfirm={() => onDeleteIntent(step.row!.name)} />
                )}
              </div>
              <div className={s.resultDesc}>{step.row.description}</div>
              <div className={s.resultMeta}>
                {step.row.sources.length > 0 && <>sources: {step.row.sources.join(', ')} · </>}
                {step.row.coverage != null && <>coverage {(step.row.coverage * 100).toFixed(0)}%</>}
                {!step.row.verified && step.row.reason && <> · {step.row.reason}</>}
              </div>
            </div>
          )}
          {step.request && step.kind === 'challenge' && <ChallengeCard request={step.request} />}
          {step.request && step.kind === 'l3' && <RequestCard request={step.request} />}
          {step.summary && <div className={s.summary}>{step.summary}</div>}
          {step.note && !step.row && step.kind !== 'summary' && <div className={s.stepNote}>{step.note}</div>}
        </div>
      )}
    </article>
  )
}

/**
 * A data-request run's step, flat: no card inside a card. A request reads top to
 * bottom as it happened - the words that led to it, the tools it took, a try that
 * was sent back, the proposal with its SQL, the agent's challenge, the outcome.
 */
function L3StepCard({ step, job, open, onToggle, requests }: {
  step: Step; job: Job; open: boolean; onToggle: () => void; requests: DataRequest[]
}) {
  const r = step.request
  const label = step.kind === 'l3' ? (r ? `${r.intent} · data request` : 'sent back to fix')
    : KIND_LABEL[step.kind]
  const status = step.kind === 'l3'
    ? (r?.refunded ? 'dropped · covered, not counted' : (L3_STATUS[step.status] ?? STATUS_LABEL[step.status]))
    : STATUS_LABEL[step.status]
  const folded = r?.challenge ? `${r.challenge.verdict} → ${r.status}` : stepSummary(step)
  return (
    <article className={`${s.step} ${s[step.kind]} ${s[step.status]}`} data-step={step.id}>
      <button className={s.stepHead} onClick={onToggle}>
        <span className={`${s.dot} ${s[step.status]}`} />
        <span className={s.kindLabel}>{label}</span>
        <span className={s.stepTitle}>{step.title}</span>
        <span className={`${s.status} ${s[step.status]}`}>{status}</span>
        <span className={s.chev}>{open ? '▾' : '▸'}</span>
      </button>
      {!open && (
        <div className={s.folded}>
          {step.tools && <ToolChips tools={step.tools} tone="dark" />}
          <span className={s.foldedMeta}>{folded}</span>
        </div>
      )}
      {open && (
        <div className={s.stepBody}>
          {step.items.map((item) => <Item key={`${item.kind}-${item.seq}`} item={item} job={job} />)}
          {r && <RequestBody request={r} />}
          {step.refused?.map((t, i) => {
            // A try with the same SQL was sent back for the order of work, not its SQL.
            const sameSql = !!r && norm(t.sql) === norm(r.sql)
            return (
              <div key={i} className={s.refusedTry}>
                <b>Sent back first</b> - nothing spent. {t.error}
                {!sameSql && (
                  <details><summary>the SQL it sent back</summary><pre className={s.src}>{t.sql}</pre></details>
                )}
              </div>
            )
          })}
          {!r && step.tools && step.tools.length > 0 && <ToolChips tools={step.tools} tone="dark" />}
          {step.kind === 'summary' && <L3Summary step={step} requests={requests} />}
        </div>
      )}
    </article>
  )
}

const norm = (sql: string) => sql.replace(/\s+/g, ' ').trim().toLowerCase()

/** A request, flat: why, what it enables, the validated SQL, the challenge. */
function RequestBody({ request: r }: { request: DataRequest }) {
  const features = r.features.split('\n').map((x) => x.replace(/^[-*\s]+/, '').trim()).filter(Boolean)
  const c = r.challenge
  return (
    <>
      <div className={s.field}><span className={s.fieldLabel}>Why</span><div className={s.fieldText}>{r.gap}</div></div>
      {features.length > 0 && (
        <div className={s.field}><span className={s.fieldLabel}>Enables</span>
          <div className={s.chips}>{features.map((f) => <code key={f}>{f}</code>)}</div></div>
      )}
      <div className={s.field}><span className={s.fieldLabel}>Reads</span>
        <div className={s.chips}>{r.tables.map((t) => <code key={t}>{t}</code>)}
          {(r.columns ?? []).map((x) => <code key={x} className={s.colChip}>{x}</code>)}</div></div>
      <SqlBlock title="SQL · validated" sql={r.sql} />
      {c && (
        <div className={s.field}><span className={s.fieldLabel}>Challenge</span>
          <div className={s.fieldText}>
            <span className={`${s.verdict} ${s['verdict_' + c.verdict]}`}>{c.verdict}</span>{' '}
            <span className={`${s.verdict} ${s['verdict_' + r.status]}`}>{r.status}</span>{' '}
            {c.reasoning}
            {c.note && <div className={s.challengeNote}>{c.note}</div>}
            {c.code && (
              <details className={s.construction}>
                <summary>its construction from current data · {c.code_ok ? 'ran on the screen rows' : 'did not run'}</summary>
                <pre className={s.src}>{c.code}</pre>
                {c.code_error && <pre className={s.err}>{c.code_error}</pre>}
              </details>
            )}
          </div>
        </div>
      )}
    </>
  )
}

/** The run's end: the agent's words, then every kept request's SQL to copy. */
function L3Summary({ step, requests }: { step: Step; requests: DataRequest[] }) {
  const [copied, setCopied] = useState(false)
  const prose = (step.summary ?? '').split('\n\nChallenge:')[0].trim()
  const kept = requests.filter((r) => r.status === 'kept')
  const dropped = requests.filter((r) => r.status === 'dropped')
  const open = requests.filter((r) => r.status === 'proposed')
  const all = kept.map((r) => `-- ${r.intent} ${r.source_name}\n${r.sql.trim()}`).join('\n\n')
  return (
    <>
      {prose && <div className={s.summary}>{prose}</div>}
      {kept.length > 0 && (
        <div className={s.keptHead}>
          <span>Validated SQL · {kept.length} kept request{kept.length > 1 ? 's' : ''}</span>
          <button className={s.copy} onClick={() => { navigator.clipboard?.writeText(all); setCopied(true) }}>
            {copied ? 'copied' : 'copy all'}</button>
        </div>
      )}
      {kept.map((r) => <SqlBlock key={r.intent} title={`${r.intent} · ${r.source_name}`} sql={r.sql} />)}
      {dropped.length > 0 && (
        <div className={s.stepNote}>Dropped - constructible from current data:{' '}
          {dropped.map((r) => `${r.intent} ${r.source_name}`).join(', ')}</div>
      )}
      {open.length > 0 && (
        <div className={s.stepNote}>Not challenged: {open.map((r) => `${r.intent} ${r.source_name}`).join(', ')}</div>
      )}
    </>
  )
}

function SqlBlock({ title, sql }: { title: string; sql: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className={s.sqlBlock}>
      <div className={s.sqlHead}>
        <span>{title}</span>
        <button className={s.copy} onClick={() => { navigator.clipboard?.writeText(sql); setCopied(true) }}>
          {copied ? 'copied' : 'copy'}</button>
      </div>
      <pre className={s.src}>{sql}</pre>
    </div>
  )
}

function Item({ item, job }: { item: TraceItem; job: Job }) {
  switch (item.kind) {
    case 'message':
      return <div className={s.message}>{item.text}</div>
    case 'tool':
      return (
        <div className={s.tool}>
          <span className={s.toolName}>{item.tool}</span>
          <span className={s.toolArgs}>{compactArgs(item.args)}</span>
        </div>
      )
    case 'skill':
      return <div className={s.tool}><span className={s.toolName}>skill</span> loaded <b>{item.name}</b></div>
    case 'code':
      return <Code card={item} />
    case 'approval':
      return <Approval card={item} job={job} />
    case 'note':
      return <div className={`${s.note} ${s[item.tone]}`}>{item.text}</div>
    case 'ideas':
      return (
        <div className={s.ideas}>
          <div className={s.ideasFocus}>this run leans on {item.focus.map((f) => <code key={f}>{f}</code>)}</div>
          {item.ideas.map((i) => {
            const proposed = i.name ? item.used?.[i.name] : undefined
            return (
              <div key={i.n} className={`${s.idea} ${proposed ? s.ideaUsed : ''}`}>
                <span className={s.ideaN}>idea_{i.n}</span>
                <div className={s.ideaBody}>
                  <div className={s.ideaHead}>
                    {i.name && <span className={s.ideaName}>{i.name}</span>}
                    {i.level && <span className={s.ideaLevel}>{i.level}</span>}
                    <code className={`${s.lens} ${item.focus.includes(i.lens) ? s.lensFocus : ''}`}>{i.lens}</code>
                    {proposed && <span className={s.ideaProposed}>→ {proposed}</span>}
                  </div>
                  <div>{i.description ?? i.hypothesis}</div>
                  {i.data && <div className={s.ideaData}>{i.data}</div>}
                </div>
              </div>
            )
          })}
        </div>
      )
  }
}

function compactArgs(args: string): string {
  try {
    return Object.entries(JSON.parse(args)).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join('  ')
  } catch {
    return args
  }
}

function Code({ card }: { card: CodeCard }) {
  const [open, setOpen] = useState(card.mode !== 'probe' || card.state === 'error')
  return (
    <div className={`${s.code} ${s[card.state]}`}>
      <button className={s.codeHead} onClick={() => setOpen(!open)}>
        <span className={`${s.dot} ${s[card.state]}`} />
        <span className={s.codeId}>{card.code_id}</span>
        <span className={`${s.tag} ${s[card.mode]}`}>{card.mode}</span>
        <span className={s.codeTitle}>{card.title}</span>
        <span className={s.codeMeta}>
          {card.state === 'running' ? 'running…' : `${card.state} · ${card.elapsed_s ?? 0}s`}
        </span>
        <span className={s.chev}>{open ? '▾' : '▸'}</span>
      </button>
      {open && <pre className={s.src}>{card.code}</pre>}
      {open && card.stdout && <pre className={s.out}>{card.stdout}</pre>}
      {card.error && <pre className={s.err}>{open ? card.error : card.error.trim().split('\n').slice(-1)[0]}</pre>}
      {open && card.result && <pre className={s.out}>{`${card.result.n_rows} rows\n${card.result.head}`}</pre>}
    </div>
  )
}

function Approval({ card, job }: { card: ApprovalCard; job: Job }) {
  const [note, setNote] = useState('')
  const [sending, setSending] = useState(false)
  const [copied, setCopied] = useState(false)
  const p = card.payload

  const answer = async (approved: boolean) => {
    setSending(true)
    try { await api.approve(job.kind, job.id, card.req_id, approved, note) } finally { setSending(false) }
  }

  return (
    <div className={`${s.approval} ${card.resolved ? s.resolved : s.pending}`}>
      <div className={s.approvalHead}>
        <span className={`eyebrow ${s.approvalKind}`}>
          {card.type === 'linkage' ? `Confirm linkage · ${p.source}` : `Approve data request · ${p.source_name}`}
        </span>
        {card.resolved
          ? <span className={card.resolved.approved ? s.yes : s.no}>
              {card.resolved.approved ? 'approved' : 'rejected'}
              {card.resolved.note ? ` - "${card.resolved.note}"` : ''}
            </span>
          : <span className={s.waiting}>waiting for you</span>}
      </div>

      {card.type === 'linkage' ? (
        <>
          <div className={s.checks}>
            <Check label="rows" value={p.rows?.toLocaleString()} />
            <Check label="match rate" value={`${((p.match_rate ?? 0) * 100).toFixed(1)}%`} />
            <Check label="point-in-time violations" value={p.point_in_time_violations}
                   bad={p.point_in_time_violations > 0} />
            <Check label="rule" value={`${p.time_column} ${p.rule === 'strict' ? '<' : '≤'} as_of`} />
          </div>
          <pre className={s.src}>{p.code}</pre>
          <pre className={s.out}>{p.head}</pre>
        </>
      ) : (
        <>
          <div className={s.gap}>{p.gap}</div>
          <div className={s.sqlHead}>
            <span>BigQuery SQL</span>
            <button className={s.copy} onClick={() => {
              navigator.clipboard?.writeText(p.sql)
              setCopied(true)
            }}>{copied ? 'copied' : 'copy'}</button>
          </div>
          <pre className={s.src}>{p.sql}</pre>
          {card.resolved?.approved && (
            <div className={s.next}>
              Run it in BigQuery, then save the result as <code>{p.source_name}.parquet</code> with
              a <code>{p.source_name}_data_sample.json</code> in the additional data folder.
            </div>
          )}
        </>
      )}

      {!card.resolved && (
        <div className={s.actions}>
          <input className={s.noteInput} value={note} placeholder="note for the agent (optional)"
                 onChange={(e) => setNote(e.target.value)} />
          <button className={s.reject} disabled={sending} onClick={() => answer(false)}>Reject</button>
          <button className={s.approve} disabled={sending} onClick={() => answer(true)}>Approve</button>
        </div>
      )}
    </div>
  )
}

/** The challenger's answer: can it be built from current data - and the attempt. */
function ChallengeCard({ request: r }: { request: DataRequest }) {
  const c = r.challenge
  if (!c) return null
  return (
    <div className={`${s.challenge} ${r.status === 'dropped' ? s.challengeDrop : s.challengeKeep}`}>
      <div className={s.requestHead}>
        <span className={s.requestName}>{r.source_name}</span>
        <span className={`${s.verdict} ${s['verdict_' + c.verdict]}`}>{c.verdict}</span>
        <span className={s.outcome}>{r.status === 'dropped'
          ? 'dropped - it can be built from current data' : 'kept - worth pulling'}</span>
      </div>
      <div className={s.requestWhy}>{c.reasoning}</div>
      {c.note && <div className={s.challengeNote}>{c.note}</div>}
      {c.columns.length > 0 && (
        <div className={s.requestFeatures}><span>Built from</span>{c.columns.map((x) => <code key={x}>{x}</code>)}</div>
      )}
      {c.code && (
        <details className={s.construction}>
          <summary>Construction from current data · {c.code_ok ? 'ran on the screen rows' : 'did not run'}</summary>
          <pre className={s.src}>{c.code}</pre>
          {c.code_error && <pre className={s.err}>{c.code_error}</pre>}
        </details>
      )}
    </div>
  )
}

/** A proposed data pull: why, what it would enable, what it reads, and the SQL. */
function RequestCard({ request: r }: { request: DataRequest }) {
  const [copied, setCopied] = useState(false)
  const features = r.features.split('\n').map((x) => x.replace(/^[-*\s]+/, '').trim()).filter(Boolean)
  return (
    <div className={`${s.request} ${r.status === 'dropped' ? s.requestDropped : ''}`}>
      <div className={s.requestHead}>
        <span className={s.requestName}>{r.source_name}</span>
        {r.status !== 'proposed' && (
          <span className={`${s.verdict} ${r.status === 'dropped' ? s.verdict_constructible : s.verdict_new}`}>{r.status}</span>
        )}
        <span className={s.requestTables}>reads {r.tables.map((t) => <code key={t}>{t}</code>)}
          {r.columns && r.columns.length > 0 && <> · columns {r.columns.map((c) => <code key={c}>{c}</code>)}</>}</span>
      </div>
      <div className={s.requestWhy}>{r.gap}</div>
      {features.length > 0 && (
        <div className={s.requestFeatures}>
          <span>Would enable</span>
          {features.map((f) => <code key={f}>{f}</code>)}
        </div>
      )}
      <div className={s.sqlHead}>
        <span>BigQuery SQL</span>
        <button className={s.copy} onClick={() => { navigator.clipboard?.writeText(r.sql); setCopied(true) }}>
          {copied ? 'copied' : 'copy'}</button>
      </div>
      <pre className={s.src}>{r.sql}</pre>
    </div>
  )
}

function Check({ label, value, bad }: { label: string; value: any; bad?: boolean }) {
  return (
    <div className={`${s.check} ${bad ? s.checkBad : ''}`}>
      <span className={s.checkLabel}>{label}</span>
      <span className={s.checkValue}>{value}</span>
    </div>
  )
}
