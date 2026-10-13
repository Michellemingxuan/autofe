import { Fragment, useEffect, useMemo, useState } from 'react'
import { api, usePolled } from '../api'
import type { LlmCallFull } from '../types'
import s from './LlmCalls.module.css'

type Props = { runId: string; onClose: () => void }

const STAGE: Record<string, string> = {
  ideas: 'Ideas', propose: 'Proposals', linkage: 'Linkage', themes: 'Themes',
}

const ms = (v: number | null) => (v === null ? '' : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${v} ms`)
const n = (v: number | null) => (v === null || v === undefined ? '' : v.toLocaleString())
const clock = (ts: number) => new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })

/** A message of the model's input or output, as it reads: its role or type, and its text. */
function Item({ item }: { item: any }) {
  const [open, setOpen] = useState(false)
  if (!item || typeof item !== 'object') return <div className={s.item}><pre>{String(item)}</pre></div>
  const type = item.type ?? (item.role ? 'message' : 'item')
  let label: string = item.role ?? type
  let body = ''
  if (type === 'function_call') {
    label = `call · ${item.name}`
    body = item.arguments ?? ''
    try { body = JSON.stringify(JSON.parse(body), null, 2) } catch { /* as given */ }
  } else if (type === 'function_call_output') {
    label = 'tool result'
    body = typeof item.output === 'string' ? item.output : JSON.stringify(item.output, null, 2)
  } else if (typeof item.content === 'string') {
    body = item.content
  } else if (Array.isArray(item.content)) {
    body = item.content.map((c: any) => c?.text ?? c?.content ?? JSON.stringify(c)).join('\n')
  } else {
    body = JSON.stringify(item, null, 2)
  }
  const long = body.length > 600
  return (
    <div className={`${s.item} ${s[label.split(' ')[0]] ?? ''}`}>
      <div className={s.itemHead}>
        <span className={s.role}>{label}</span>
        <span className={s.size}>{body.length.toLocaleString()} chars</span>
        {long && <button className={s.more} onClick={() => setOpen((v) => !v)}>{open ? 'less' : 'all'}</button>}
      </div>
      <pre className={s.body}>{long && !open ? `${body.slice(0, 600)}…` : body}</pre>
    </div>
  )
}

function Payload({ value }: { value: LlmCallFull['input_json'] }) {
  if (!value) return <div className={s.none}>nothing recorded</div>
  if (!Array.isArray(value)) return <div className={s.none}>{(value as any)._elided} ({n((value as any).size)} chars)</div>
  return <>{value.map((it, i) => <Item key={i} item={it} />)}</>
}

/** One call in full: the brief it ran under, every message it was given, and its answer. */
function Detail({ id }: { id: number }) {
  const [call, setCall] = useState<LlmCallFull | null>(null)
  const [brief, setBrief] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    setCall(null)
    api.llmCall(id).then(setCall).catch((e) => setError(String(e.message ?? e)))
  }, [id])
  if (error) return <div className={s.none}>{error}</div>
  if (!call) return <div className={s.none}>Loading…</div>
  const input = Array.isArray(call.input_json) ? call.input_json : []
  return (
    <div className={s.detail}>
      <div className={s.detailHead}>
        Call {call.call} · {STAGE[call.stage] ?? call.stage}{call.round ? ` · round ${call.round}` : ''} ·{' '}
        {call.model} · {ms(call.duration_ms)} · {n(call.input_tokens)} in{call.cached_tokens ? ` (${n(call.cached_tokens)} cached)` : ''}
        {' '}/ {n(call.output_tokens)} out{call.tokens_estimated ? ' (estimated)' : ''}
      </div>
      <section>
        <h4>
          Instructions <span>{n(call.system_prompt.length)} chars - the run's brief</span>
          <button className={s.more} onClick={() => setBrief((v) => !v)}>{brief ? 'hide' : 'show'}</button>
        </h4>
        {brief && <pre className={s.brief}>{call.system_prompt}</pre>}
      </section>
      <section>
        <h4>Input <span>{input.length} messages - exactly what the model saw (old tool results cut)</span></h4>
        <Payload value={call.input_json} />
      </section>
      <section>
        <h4>Output <span>{call.outcome === 'ok' ? 'its answer' : 'the call never returned'}</span></h4>
        <Payload value={call.output_json} />
      </section>
    </div>
  )
}

/** Every model call of a run, in order: the lifecycle under the trace. */
export function LlmCalls({ runId, onClose }: Props) {
  const [calls] = usePolled(() => api.llmCalls(runId), [runId], 4000)
  const [selected, setSelected] = useState<number | null>(null)
  // The standalone trace viewer, when the server started one beside it.
  const [viewer, setViewer] = useState<string | null>(null)
  useEffect(() => { api.workspace().then((w) => setViewer(w.trace_viewer ?? null)).catch(() => {}) }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const totals = useMemo(() => {
    const all = calls ?? []
    return {
      n: all.length,
      input: all.reduce((t, c) => t + (c.input_tokens ?? 0), 0),
      output: all.reduce((t, c) => t + (c.output_tokens ?? 0), 0),
      seconds: all.reduce((t, c) => t + (c.duration_ms ?? 0), 0) / 1000,
      estimated: all.some((c) => c.tokens_estimated),
    }
  }, [calls])

  return (
    <div className={s.backdrop} onClick={onClose}>
      <section className={s.panel} onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Model calls">
        <header className={s.head}>
          <div className={s.stats}>
            <span className="eyebrow">Model calls</span>
            <div className={s.stat}><b>{totals.n}</b> calls</div>
            <div className={s.stat}><b>{n(totals.input)}</b> input tokens</div>
            <div className={s.stat}><b>{n(totals.output)}</b> output tokens{totals.estimated ? ' (estimated)' : ''}</div>
            <div className={s.stat}><b>{totals.seconds.toFixed(0)}</b> s in the model</div>
          </div>
          <div className={s.headRight}>
            {viewer && (
              <a className={s.open} href={`${viewer}/run/${runId}`} target="_blank" rel="noreferrer"
                 title="The same calls in the standalone trace viewer, in a new tab">Open in trace viewer ↗</a>
            )}
            <button className={s.close} onClick={onClose} aria-label="Close">×</button>
          </div>
        </header>
        <div className={s.body2}>
          <div className={s.list}>
            {calls === null && <div className={s.none}>Loading…</div>}
            {calls !== null && calls.length === 0 && (
              <div className={s.none}>No model calls recorded for this run - runs from before the trace store have none.</div>
            )}
            {(calls ?? []).map((c, i, all) => {
              const head = i === 0 || all[i - 1].stage !== c.stage || all[i - 1].round !== c.round
              return (
                <Fragment key={c.id}>
                  {head && (
                    <div className={s.stageHead}>
                      {STAGE[c.stage] ?? c.stage}{c.round ? ` · round ${c.round}` : ''}
                    </div>
                  )}
                  <button className={`${s.row} ${selected === c.id ? s.rowOn : ''}`} onClick={() => setSelected(c.id)}>
                    <span className={s.callN}>{c.call}</span>
                    <span className={s.rowMain}>
                      <span className={s.rowWhat}>
                        {c.tool_calls.length ? c.tool_calls.map((t) => <code key={t}>{t}</code>) : <em>{c.outcome === 'ok' ? 'text answer' : c.outcome}</em>}
                      </span>
                      <span className={s.rowMeta}>
                        {clock(c.started_at)} · {ms(c.duration_ms)} · {n(c.input_tokens)} → {n(c.output_tokens)} tokens
                        {c.outcome !== 'ok' && <b className={s.bad}> · {c.outcome}</b>}
                      </span>
                    </span>
                  </button>
                </Fragment>
              )
            })}
          </div>
          <div className={s.side}>
            {selected === null ? <div className={s.none}>Pick a call to see its input and output.</div> : <Detail id={selected} />}
          </div>
        </div>
      </section>
    </div>
  )
}
