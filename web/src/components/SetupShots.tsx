import { useState } from 'react'
import { api } from '../api'
import type { PathCheck, SetupBlock, ShotCategory, Workspace } from '../types'
import { ConfirmButton } from './ConfirmButton'
import { Help, PathOrUpload, Tick, UploadButton, usePathCheck } from './SetupFields'
import s from './SetupPage.module.css'

type Values = Record<string, unknown>


/** Take a category off the agent's list: the clustering file, or one spec. */
export async function removeShot(cat: ShotCategory, block: SetupBlock) {
  const specPaths = (block.fields.find((f) => f.key === 'discovery.shot_spec_paths')?.value as string[]) ?? []
  await api.applySetup(cat.kind === 'clustering'
    ? { 'discovery.few_shot_path': '' }
    : { 'discovery.shot_spec_paths': specPaths.filter((p) => p !== cat.path) })
}

/**
 * The shots, in the order the agent reads them: the clustering shots first,
 * then your own categories appended after - and, last, what is available.
 */
export function ShotsBody({ block, values, set, checks, workspace, onApplied }: {
  block: SetupBlock; values: Values; set: (k: string, v: unknown) => void
  checks: PathCheck; workspace: Workspace | null; onApplied: () => void
}) {
  const cats = workspace?.shots ?? []
  const clustering = cats.find((c) => c.kind === 'clustering')
  const fileField = block.fields.find((f) => f.key === 'discovery.few_shot_path')
  return (
    <>
      <section className={s.part}>
        <h3 className={s.partTitle}>Clustering shots</h3>
        <p className={s.partLead}>
          {clustering
            ? <>Using <b>{clustering.found}</b> rows in <b>{clustering.batches}</b> batch{clustering.batches > 1 ? 'es' : ''},
                one batch per discovery run.</>
            : 'Not set. Generate them from the screen rows, or use a file the prepare step already wrote.'}
        </p>
        <div className={s.options}>
          <GenerateTile exists={!!clustering} onDone={onApplied} />
          <div className={s.option}>
            <div className={s.optionTitle}>Use a precached file</div>
            <p className={s.optionText}>A CSV of labelled rows with a <code>batch</code> column, as
              <code> prepare.ipynb</code> writes it. Apply the block to use it.</p>
            {fileField && (
              <PathOrUpload field={{ ...fileField, label: 'File' }} value={String(values[fileField.key] ?? '')}
                            onChange={(v) => set(fileField.key, v)} checks={checks}
                            dirty={values[fileField.key] !== fileField.value} />
            )}
          </div>
        </div>
      </section>

      <section className={s.part}>
        <h3 className={s.partTitle}>Your own categories</h3>
        <p className={s.partLead}>
          A group of example rows with a story - what they are and why they matter. Read after
          the clustering shots, in order. What is loaded is listed on the right.
        </p>
        <AddCategory onAdded={onApplied} />
      </section>
    </>
  )
}

function GenerateTile({ exists, onDone }: { exists: boolean; onDone: () => void }) {
  const [rows, setRows] = useState(32)
  const [batches, setBatches] = useState(4)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  return (
    <div className={s.option}>
      <div className={s.optionTitle}>{exists ? 'Regenerate' : 'Generate'} from the screen rows</div>
      <p className={s.optionText}>KMeans per class picks one row per cluster, so every class appears
        and the rows cover the table. Each batch takes the next-closest rows.</p>
      <div className={s.genRow}>
        <label><input type="number" min={2} max={400} value={rows}
                      onChange={(e) => setRows(Math.max(2, Number(e.target.value) || 2))} /> rows</label>
        <span className={s.times}>×</span>
        <label><input type="number" min={1} max={20} value={batches}
                      onChange={(e) => setBatches(Math.max(1, Number(e.target.value) || 1))} /> batches</label>
        <button className={s.primarySmall} disabled={busy} onClick={async () => {
          setBusy(true)
          setError(null)
          try { await api.generateShots(rows, batches); onDone() } catch (e: any) { setError(e.message) }
          finally { setBusy(false) }
        }}>{busy ? 'Clustering…' : exists ? 'Regenerate' : 'Generate'}</button>
      </div>
      {error && <div className={s.error}>{error}</div>}
    </div>
  )
}

function AddCategory({ onAdded }: { onAdded: () => void }) {
  const [open, setOpen] = useState(false)
  const [form, setForm] = useState({ name: '', context: '', table: '', rotate: false, batch: 8 })
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const checks = usePathCheck([form.table])

  if (!open) return <button className={s.addButton} onClick={() => setOpen(true)}>+ Add a category</button>

  const add = async () => {
    setBusy(true)
    setError(null)
    try {
      await api.addShotCategory({ name: form.name, context: form.context, table: form.table,
                                  rotate: form.rotate, batch_size: form.rotate ? form.batch : undefined })
      setForm({ name: '', context: '', table: '', rotate: false, batch: 8 })
      setOpen(false)
      onAdded()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className={s.addCat}>
      <div className={s.catForm}>
        <div className={s.formRow}>
          <label className={s.field}><span className={s.fieldLabel}>Name</span>
            <input value={form.name} placeholder="High utilisation"
                   onChange={(e) => setForm({ ...form, name: e.target.value })} /></label>
          <div className={s.field}>
            <span className={s.fieldLabel}>Example table
              <Help text="A CSV of example rows. Rows that carry the id column are checked: any from the screen's scored rows are left out, so held-out data never reaches the agent." /></span>
            <div className={s.inputRow}>
              <input className={s.mono} value={form.table} placeholder="path to a .csv, or upload"
                     onChange={(e) => setForm({ ...form, table: e.target.value })} />
              <UploadButton kind="shot_table" onUploaded={(p) => setForm((f) => ({ ...f, table: p }))} />
              <Tick path={form.table} checks={checks} />
            </div>
          </div>
        </div>
        <label className={s.field}><span className={s.fieldLabel}>Context</span>
          <textarea rows={2} value={form.context}
                    placeholder="What these rows have in common, and why they matter for default risk"
                    onChange={(e) => setForm({ ...form, context: e.target.value })} /></label>
        <div className={s.formFoot}>
          <span className={s.fieldLabel}>Each run sees</span>
          <div className={s.segment} role="radiogroup">
            <button className={!form.rotate ? s.segOn : ''} onClick={() => setForm({ ...form, rotate: false })}>all rows</button>
            <button className={form.rotate ? s.segOn : ''} onClick={() => setForm({ ...form, rotate: true })}>a rotating batch</button>
          </div>
          {form.rotate && (
            <label className={s.batchLabel}>
              <input type="number" min={1} value={form.batch}
                     onChange={(e) => setForm({ ...form, batch: Math.max(1, Number(e.target.value) || 1) })} /> per batch</label>
          )}
          <span className={s.grow} />
          <button className={s.secondary} onClick={() => { setOpen(false); setError(null) }}>Cancel</button>
          <button className={s.primary} disabled={!form.name.trim() || !form.table.trim() || busy} onClick={add}>
            {busy ? 'Adding…' : 'Add category'}</button>
        </div>
        {error && <div className={s.error}>{error}</div>}
      </div>
    </div>
  )
}

const KIND_LABEL: Record<ShotCategory['kind'], string> = {
  clustering: 'clustering', ids: 'ids', table: 'table', error: 'error',
}

/** Every category: its rows, class mix and rotation, then the totals. */
export function ShotOverview({ cats, target, onRemove }: {
  cats: ShotCategory[]; target: string; onRemove: (c: ShotCategory) => Promise<void>
}) {
  const [open, setOpen] = useState<string | null>(null)
  if (!cats.length) return <p className={s.ovEmpty}>No shots yet.</p>
  const classes = [...new Set(cats.flatMap((c) => Object.keys(c.classes)))].sort()
  const total = cats.reduce((n, c) => n + c.found, 0)
  const totals = Object.fromEntries(classes.map((k) => [k, cats.reduce((n, c) => n + (c.classes[k] ?? 0), 0)]))
  return (
    <div className={s.shotList}>
      {cats.map((c) => (
        <div key={c.key} className={s.shotItem}>
          <div className={s.shotTop}>
            <button className={s.catName} onClick={() => setOpen(open === c.key ? null : c.key)}>
              <span className={s.chev}>{open === c.key ? '▾' : '▸'}</span>
              <span>{c.name}</span>
              <span className={`${s.kind} ${s['kind_' + c.kind]}`}>{KIND_LABEL[c.kind]}</span>
            </button>
            <span className={s.catNum}>{c.found} rows</span>
          </div>
          <ClassBar classes={c.classes} order={classes} />
          <div className={s.shotMeta}>
            {c.rotate
              ? <><span className={s.batchBlocks}>{c.per_batch.map((n, i) => <i key={i} style={{ flex: n }} title={`batch ${i + 1}: ${n} rows`} />)}</span>
                  <span>1 of {c.batches} batches per run</span></>
              : <span>all rows every run</span>}
            {c.n_missing > 0 && <span className={s.warnText} title={c.missing.join(', ')}>{c.n_missing} left out</span>}
          </div>
          {open === c.key && (
            <div className={s.catDetail}>
              <p className={c.kind === 'error' ? s.warnText : undefined}>
                {c.kind === 'error' ? `Not loaded: ${c.context}` : c.context || 'No context given.'}</p>
              {c.path && <code>{c.path}</code>}
              {c.path && <ConfirmButton label="Remove" confirm="Remove this category?" onConfirm={() => onRemove(c)} />}
            </div>
          )}
        </div>
      ))}
      {cats.length > 1 && (
        <div className={s.shotTotal}>
          <span><b>{total}</b> rows across {cats.length} categories · {target}</span>
          <ClassBar classes={totals} order={classes} />
        </div>
      )}
    </div>
  )
}

const CLASS_TONES = ['var(--j-ink-mute)', 'var(--j-accent)', '#7c5cc4', '#c9a24a', '#2f9e8f']

function ClassBar({ classes, order }: { classes: Record<string, number>; order: string[] }) {
  const total = Object.values(classes).reduce((a, b) => a + b, 0)
  if (!total) return <span className={s.hint}>no target column</span>
  return (
    <span className={s.classMix}>
      <span className={s.classBar}>
        {order.map((k, i) => classes[k] ? (
          <i key={k} style={{ flex: classes[k], background: CLASS_TONES[i % CLASS_TONES.length] }}
             title={`${k}: ${classes[k]}`} />
        ) : null)}
      </span>
      <span className={s.classText}>
        {order.filter((k) => classes[k]).map((k) => `${k}: ${classes[k]}`).join('  ')}
      </span>
    </span>
  )
}
