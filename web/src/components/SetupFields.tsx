import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { PathCheck, SetupField } from '../types'
import s from './SetupPage.module.css'

/** Whether the paths on screen exist - asked as the user types, debounced. */
export function usePathCheck(paths: string[]): PathCheck {
  const [checked, setChecked] = useState<PathCheck>({})
  const key = paths.filter(Boolean).join('\n')
  useEffect(() => {
    if (!key) return
    const timer = setTimeout(() => {
      api.checkPaths(key.split('\n')).then((r) => setChecked((c) => ({ ...c, ...r }))).catch(() => {})
    }, 350)
    return () => clearTimeout(timer)
  }, [key])
  return checked
}

const size = (bytes: number | null) =>
  bytes == null ? '' : bytes > 1e9 ? `${(bytes / 1e9).toFixed(1)} GB`
    : bytes > 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1e3))} KB`

/** A name inside the folder as a full path; absolute names and no folder stay as they are. */
export const inFolder = (folder: string, name: string) =>
  !name || name.startsWith('/') || !folder ? name : `${folder.replace(/\/$/, '')}/${name}`

/** A path as a name inside the folder, when it is in it. */
export const relTo = (folder: string, path: string) => {
  const f = folder.replace(/\/$/, '')
  return f && path.startsWith(f + '/') ? path.slice(f.length + 1) : path
}

/** A ✓ or ✗ beside a path, with the file's size when it is there. */
export function Tick({ path, checks }: { path: string; checks: PathCheck }) {
  if (!path) return <span className={s.tickEmpty} />
  const c = checks[path]
  if (!c) return <span className={s.tickPending}>…</span>
  return c.exists
    ? <span className={s.tickOk} title={path}>✓ {c.dir ? 'folder' : size(c.bytes)}</span>
    : <span className={s.tickBad} title={path}>✗ not found</span>
}

/** A "?" that shows the help on hover or keyboard focus - the form stays quiet. */
export function Help({ text }: { text: string }) {
  return (
    <span className={s.help} tabIndex={0} aria-label={text}>
      ?<span className={s.helpTip} role="tooltip">{text}</span>
    </span>
  )
}

/** The label; its help is behind a "?". */
export function FieldLabel({ field, dirty }: { field: SetupField; dirty?: boolean }) {
  return (
    <span className={s.fieldLabel}>
      <span className={s.labelText}>{field.label}</span>
      {(dirty || field.changed) && <span className={s.changed}>{dirty ? 'edited' : 'changed'}</span>}
      {field.help && <Help text={field.help} />}
    </span>
  )
}

/** Shown only when the value differs from the config file - what it was there. */
export function DefaultHint({ field, value }: { field: SetupField; value: unknown }) {
  const d = Array.isArray(field.default) ? field.default.join(', ') : String(field.default ?? '')
  const v = Array.isArray(value) ? value.join(', ') : String(value ?? '')
  if (!d || v === d) return null
  return <span className={s.fieldHelp}>config file: <code>{d}</code></span>
}

/** Upload a small file; the setting then points at the stored copy. */
export function UploadButton({ kind, onUploaded, label = 'Upload' }: {
  kind: string; onUploaded: (path: string) => void; label?: string
}) {
  const input = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  return (
    <>
      <button type="button" className={s.upload} disabled={busy}
              onClick={() => input.current?.click()}>{busy ? 'Uploading…' : label}</button>
      <input ref={input} type="file" hidden onChange={async (e) => {
        const file = e.target.files?.[0]
        e.target.value = ''
        if (!file) return
        setBusy(true)
        setError(null)
        try {
          onUploaded((await api.upload(kind, file)).path)
        } catch (err: any) {
          setError(err.message)
        } finally {
          setBusy(false)
        }
      }} />
      {error && <span className={s.inlineError}>{error}</span>}
    </>
  )
}

/** A small file: type its path, or upload it. With a folder, a name inside it. */
export function PathOrUpload({ field, value, onChange, checks, dirty, folder = '' }: {
  field: SetupField; value: string; onChange: (v: string) => void; checks: PathCheck; dirty: boolean
  folder?: string
}) {
  return (
    <div className={s.field}>
      <FieldLabel field={field} dirty={dirty} />
      <div className={s.inputRow}>
        <input className={s.mono} value={relTo(folder, value)} placeholder="path, or upload →"
               title={value} onChange={(e) => onChange(inFolder(folder, e.target.value))} />
        {field.upload && <UploadButton kind={field.upload} onUploaded={onChange} />}
        <Tick path={value} checks={checks} />
      </div>
      <DefaultHint field={field} value={value} />
    </div>
  )
}

/** A list of small files: each by path or upload, removable. With a folder,
 * a typed path may be relative to it. */
export function FilesList({ field, value, onChange, checks, dirty, folder = '' }: {
  field: SetupField; value: string[]; onChange: (v: string[]) => void; checks: PathCheck; dirty: boolean
  folder?: string
}) {
  const [adding, setAdding] = useState('')
  const add = (path: string) => {
    if (path && !value.includes(path)) onChange([...value, path])
    setAdding('')
  }
  return (
    <div className={s.field}>
      <FieldLabel field={field} dirty={dirty} />
      {value.length === 0 && <span className={s.fieldHelp}>none yet</span>}
      {value.map((p) => (
        <div key={p} className={s.listItem}>
          <span className={s.listPath} title={p}>{p.split('/').pop()}</span>
          <span className={s.listDir} title={p}>{relTo(folder, p)}</span>
          <Tick path={p} checks={checks} />
          <button className={s.remove} onClick={() => onChange(value.filter((x) => x !== p))}
                  title="remove from the list (the file stays)">×</button>
        </div>
      ))}
      <div className={s.inputRow}>
        <input className={s.mono} value={adding}
               placeholder={folder ? 'add a path in the folder, or from / …' : 'add a path…'}
               onChange={(e) => setAdding(e.target.value)}
               onKeyDown={(e) => { if (e.key === 'Enter') add(inFolder(folder, adding.trim())) }} />
        <button className={s.secondary} disabled={!adding.trim()}
                onClick={() => add(inFolder(folder, adding.trim()))}>Add</button>
        {field.upload && <UploadButton kind={field.upload} onUploaded={add} />}
      </div>
    </div>
  )
}

/** A plain value or a big file by path - with a folder, a name inside it. */
export function PlainField({ field, value, onChange, checks, dirty, folder = '' }: {
  field: SetupField; value: string; onChange: (v: string) => void; checks: PathCheck; dirty: boolean
  folder?: string
}) {
  const path = field.kind === 'path'
  return (
    <div className={s.field}>
      <FieldLabel field={field} dirty={dirty} />
      {field.kind === 'longtext'
        ? <textarea rows={3} value={value} onChange={(e) => onChange(e.target.value)} />
        : (
          <div className={s.inputRow}>
            <input className={path ? s.mono : ''} value={path ? relTo(folder, value) : value}
                   title={path ? value : undefined}
                   onChange={(e) => onChange(path ? inFolder(folder, e.target.value) : e.target.value)} />
            {field.kind === 'path' && <Tick path={value} checks={checks} />}
          </div>
        )}
      <DefaultHint field={field} value={value} />
    </div>
  )
}

/** A form value in the shape the config holds, so edits compare like for like. */
export function normalise(field: SetupField, value: unknown): unknown {
  switch (field.kind) {
    case 'number':
      return value === '' || value == null ? '' : Number(value)
    case 'bool':
      return value === true || value === 'true'
    case 'numbers':
      return (Array.isArray(value) ? value : String(value ?? '').split(/[\s,;]+/))
        .filter((v) => String(v).trim() !== '').map(Number)
    case 'files':
      return Array.isArray(value) ? value : []
    default:
      return String(value ?? '')
  }
}

/** Numbers, switches, choices and lists of numbers - the evaluation settings. */
export function TypedField({ field, value, onChange, dirty }: {
  field: SetupField; value: unknown; onChange: (v: unknown) => void; dirty: boolean
}) {
  let control: React.ReactNode
  if (field.kind === 'bool') {
    const on = normalise(field, value) as boolean
    control = (
      <button type="button" role="switch" aria-checked={on}
              className={`${s.switch} ${on ? s.switchOn : ''}`} onClick={() => onChange(!on)}>
        <span className={s.knob} />{on ? 'On' : 'Off'}
      </button>
    )
  } else if (field.kind === 'select') {
    control = (
      <div className={s.segment} role="radiogroup">
        {(field.options ?? []).map((o) => (
          <button key={o} type="button" role="radio" aria-checked={value === o}
                  className={value === o ? s.segOn : ''} onClick={() => onChange(o)}>{o}</button>
        ))}
      </div>
    )
  } else if (field.kind === 'numbers') {
    control = <input className={s.mono} value={Array.isArray(value) ? value.join(', ') : String(value ?? '')}
                     onChange={(e) => onChange(e.target.value)} />
  } else {
    control = <input className={s.mono} type="number" step={field.int ? 1 : 'any'}
                     value={String(value ?? '')} onChange={(e) => onChange(e.target.value)} />
  }
  return (
    <div className={s.field}>
      <FieldLabel field={field} dirty={dirty} />
      {control}
      <DefaultHint field={field} value={value} />
    </div>
  )
}
