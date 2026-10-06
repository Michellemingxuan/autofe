import { useEffect, useMemo, useState } from 'react'
import { api, streamUrl, useEventStream } from '../api'
import { deriveRun } from '../derive'
import type { PathCheck, SetupField, Source } from '../types'
import { SOURCE_STATE } from './Rail'
import { Trace } from './Trace'
import { ConfirmButton } from './ConfirmButton'
import { Tick, UploadButton, usePathCheck } from './SetupFields'
import s from './SetupPage.module.css'

// Additional data: a source's row (state, columns, linkage), adding one by
// path or upload, and the agent's live linkage proposal.

export function SourceRow({ source, busy, onPropose, onRemove }: {
  source: Source; busy: boolean; onPropose: () => Promise<void>; onRemove: () => Promise<void>
}) {
  const [open, setOpen] = useState(false)
  const [code, setCode] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (open && source.state === 'linked') api.linkageCode(source.name).then((r) => setCode(r.code)).catch(() => {})
  }, [open, source.state, source.name])

  return (
    <div className={`${s.source} ${open ? s.sourceOpen : ''}`}>
      <div className={s.sourceTop}>
        <button className={s.sourceName} onClick={() => setOpen(!open)}>
          <span className={s.chev}>{open ? '▾' : '▸'}</span>{source.name}
          {source.registered && <span className={s.registered}>by path</span>}
        </button>
        <span className={`${s.badge} ${s[source.state]}`} title={SOURCE_STATE[source.state].hint}>
          {SOURCE_STATE[source.state].label}
        </span>
        {source.usable && (
          <button className={source.state === 'linked' ? s.secondary : s.primarySmall}
                  disabled={busy} title={busy ? 'An agent job is running' : undefined}
                  onClick={async () => {
                    setError(null)
                    try { await onPropose() } catch (e: any) { setError(e.message) }
                  }}>
            {source.state === 'linked' ? 'Re-propose' : 'Propose linkage'}
          </button>
        )}
        {source.registered && (
          <ConfirmButton label="Remove" confirm="Forget this source?" onConfirm={onRemove}
                         title="Forget the registration; the files stay" />
        )}
      </div>
      {error && <div className={s.error}>{error}</div>}
      {open && (
        <div className={s.sourceBody}>
          <div className={s.hint}>data: <code>{source.data ?? 'not found - schema only'}</code></div>
          <table className={s.columns}>
            <tbody>
              {Object.entries(source.columns).map(([c, d]) => (
                <tr key={c}><td className={s.mono}>{c}</td><td>{d}</td></tr>
              ))}
            </tbody>
          </table>
          {code && <><div className={s.hint}>confirmed linkage</div><pre className={s.code}>{code}</pre></>}
        </div>
      )}
    </div>
  )
}

export function AddSource({ checks, onAdded }: { checks: PathCheck; onAdded: () => void }) {
  const [open, setOpen] = useState(false)
  const [form, setForm] = useState({ name: '', schema: '', data: '' })
  const [error, setError] = useState<string | null>(null)
  const extra = usePathCheck([form.schema, form.data])
  const all = { ...checks, ...extra }
  if (!open) return <button className={s.link} onClick={() => setOpen(true)}>+ Add a source</button>
  const schemaField = { label: 'Description (sample JSON)', help: '{column: [description, [sample values]]}' } as SetupField
  return (
    <div className={s.addForm}>
      <label className={s.field}><span className={s.fieldLabel}>Name</span>
        <input value={form.name} placeholder="bureau" onChange={(e) => setForm({ ...form, name: e.target.value })} /></label>
      <div className={s.field}>
        <span className={s.fieldLabel}>{schemaField.label}</span>
        <div className={s.inputRow}>
          <input className={s.mono} value={form.schema} placeholder="path, or upload →"
                 onChange={(e) => setForm({ ...form, schema: e.target.value })} />
          <UploadButton kind="source_schema" onUploaded={(p) => setForm((f) => ({ ...f, schema: p }))} />
          <Tick path={form.schema} checks={all} />
        </div>
        <span className={s.fieldHelp}>{schemaField.help}</span>
      </div>
      <div className={s.field}>
        <span className={s.fieldLabel}>Data (parquet or csv)</span>
        <div className={s.inputRow}>
          <input className={s.mono} value={form.data} placeholder="/path/bureau.parquet - optional"
                 onChange={(e) => setForm({ ...form, data: e.target.value })} />
          <Tick path={form.data} checks={all} />
        </div>
        <span className={s.fieldHelp}>Big files are registered where they lie - by path, never uploaded or copied.</span>
      </div>
      {error && <div className={s.error}>{error}</div>}
      <div className={s.actions}>
        <button className={s.secondary} onClick={() => { setOpen(false); setError(null) }}>Cancel</button>
        <button className={s.primary} disabled={!form.name || !form.schema} onClick={async () => {
          try {
            await api.addSource(form.name.trim(), form.schema.trim(), form.data.trim())
            setForm({ name: '', schema: '', data: '' })
            setOpen(false)
            onAdded()
          } catch (e: any) { setError(e.message) }
        }}>Add source</button>
      </div>
    </div>
  )
}

export function LinkageJob({ jobId, onClose }: { jobId: string; onClose: () => void }) {
  const { events } = useEventStream(streamUrl.linkage(jobId))
  const view = useMemo(() => deriveRun(events), [events])
  const live = view.status === 'running' || view.status === 'waiting'
  return (
    <section className={s.job}>
      <div className={s.jobHead}>
        <span className={`eyebrow ${s.jobTitle}`}>Linkage proposal · {view.direction.replace(/^Link /, '')}</span>
        <span className={`${s.jobState} ${s[view.status]}`}>
          {view.status === 'waiting' ? 'waiting for you' : live ? 'agent working' : view.status}
        </span>
        {live
          ? <button className={s.ghost} onClick={() => api.cancel('linkage', jobId)}>Stop</button>
          : <button className={s.ghost} onClick={onClose}>Close</button>}
      </div>
      <div className={s.jobTrace}>
        <Trace jobKind="linkage" jobId={jobId} view={view} />
      </div>
    </section>
  )
}
