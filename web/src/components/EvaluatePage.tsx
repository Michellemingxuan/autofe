import { useEffect, useMemo, useState } from 'react'
import { api, usePolled } from '../api'
import type { PoolFeature } from '../types'
import { ConfirmButton } from './ConfirmButton'
import { signed } from './EvalPanel'
import s from './EvaluatePage.module.css'

/**
 * Step 3, the feature set. Every verified feature from every direction, in one
 * pool. Pick any - singly, or grouped into combinations - and the pipeline trains
 * base + each on the full splits and scores them on test, the out-of-time
 * hold-out. The run and its results open on the Results page.
 */
export function EvaluatePage({ onStarted }: { onStarted: (id: string) => void }) {
  const [pool, refreshPool] = usePolled(api.features, [], 10000)
  const [picked, setPicked] = useState<string[]>([])
  const [combos, setCombos] = useState<Record<string, string[]>>({})
  const [building, setBuilding] = useState<string[]>([])
  const [comboName, setComboName] = useState('')
  const [opened, setOpened] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [query, setQuery] = useState('')

  const byKey = useMemo(() => Object.fromEntries((pool ?? []).map((f) => [f.key, f])), [pool])
  const shown = useMemo(() => (pool ?? []).filter((f) => matches(query,
    [f.name, f.description, f.direction, f.level, f.run_id, f.code, ...f.sources])), [pool, query])
  const groups = useMemo(() => {
    const out: { run_id: string; direction: string; features: PoolFeature[] }[] = []
    for (const f of shown) {
      let g = out.find((x) => x.run_id === f.run_id)
      if (!g) out.push(g = { run_id: f.run_id, direction: f.direction, features: [] })
      g.features.push(f)
    }
    return out
  }, [shown])

  // A feature removed from the pool leaves the selection and any combination.
  useEffect(() => {
    if (!pool) return
    const alive = new Set(pool.map((f) => f.key))
    setPicked((p) => p.filter((k) => alive.has(k)))
    setBuilding((b) => b.filter((k) => alive.has(k)))
    setCombos((c) => Object.fromEntries(Object.entries(c).filter(([, ks]) => ks.every((k) => alive.has(k)))))
  }, [pool])

  const toggle = (key: string) => {
    setPicked((p) => (p.includes(key) ? p.filter((k) => k !== key) : [...p, key]))
    setBuilding((b) => b.filter((k) => k !== key))
  }
  const addCombo = () => {
    const name = comboName.trim().replace(/\W+/g, '_') || `combo_${Object.keys(combos).length + 1}`
    setCombos({ ...combos, [name]: building })
    setBuilding([])
    setComboName('')
  }
  const run = async () => {
    setError(null)
    setStarting(true)
    try {
      const { eval_id } = await api.evaluate(picked, combos)
      onStarted(eval_id)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setStarting(false)
    }
  }

  return (
    <div className={s.page}>
      <header className={s.header}>
        <span className={`eyebrow ${s.eyebrow}`}>Step 3 · Evaluate</span>
        <h1 className={s.title}>Feature set</h1>
      </header>

      <div className={s.body}>
        <section className={s.pool}>
          <div className={s.panelHead}>
            <span className={`eyebrow ${s.panelTitle}`}>Feature pool</span>
            <span className={s.meta}>{pool?.length ?? 0} verified · {groups.length} directions · click a feature for its code</span>
          </div>
          <SearchBox value={query} onChange={setQuery} shown={shown.length} total={pool?.length ?? 0}
                     placeholder="Search name, description, direction, source or code" />
          <div className={s.scroll}>
            {pool && pool.length === 0 && (
              <div className={s.empty}>No verified features yet - run a direction in Discover.</div>
            )}
            {pool && pool.length > 0 && shown.length === 0 && (
              <div className={s.empty}>No feature matches “{query.trim()}”.</div>
            )}
            {groups.map((g) => (
              <div key={g.run_id} className={s.group}>
                <div className={s.groupHead}>
                  <span className={s.groupDirection}>{g.direction}</span>
                  <button className={s.link} onClick={() => {
                    const keys = g.features.filter((f) => !f.missing_linkage.length).map((f) => f.key)
                    const all = keys.every((k) => picked.includes(k))
                    setPicked(all ? picked.filter((k) => !keys.includes(k))
                                  : [...picked, ...keys.filter((k) => !picked.includes(k))])
                  }}>select all</button>
                </div>
                {g.features.map((f) => (
                  <PoolRow key={f.key} feature={f} picked={picked.includes(f.key)}
                           open={opened === f.key}
                           onPick={() => toggle(f.key)}
                           onOpen={() => setOpened(opened === f.key ? null : f.key)}
                           onRemove={async () => { await api.deleteIntent(f.run_id, f.name); refreshPool() }} />
                ))}
              </div>
            ))}
          </div>
        </section>

        <div className={s.side}>
          <section className={s.card}>
            <div className={`eyebrow ${s.panelTitle}`}>Evaluation set</div>
            {picked.length === 0
              ? <div className={s.hint}>Tick features in the pool - from any direction.</div>
              : (
                <div className={s.chips}>
                  {picked.map((k) => (
                    <label key={k} className={`${s.chip} ${building.includes(k) ? s.chipOn : ''}`}
                           title="tick to put it in a combination">
                      <input type="checkbox" checked={building.includes(k)}
                             onChange={() => setBuilding((b) => b.includes(k) ? b.filter((x) => x !== k) : [...b, k])} />
                      {byKey[k]?.name ?? k}
                    </label>
                  ))}
                </div>
              )}
            <div className={s.comboRow}>
              <input className={s.input} value={comboName} placeholder="combination name"
                     onChange={(e) => setComboName(e.target.value)} />
              <button className={s.secondary} disabled={building.length < 2} onClick={addCombo}>
                Combine {building.length >= 2 ? building.length : ''}
              </button>
            </div>
            <div className={s.hint}>Tick two or more chips to evaluate them together as base + the group.</div>
            {Object.entries(combos).map(([name, keys]) => (
              <div key={name} className={s.combo}>
                <span className={s.comboName}>⊕ {name}</span>
                <span className={s.comboMembers}>base + {keys.map((k) => byKey[k]?.name ?? k).join(' + ')}</span>
                <button className={s.remove} onClick={() => {
                  const { [name]: _, ...rest } = combos
                  setCombos(rest)
                }}>×</button>
              </div>
            ))}
            {error && <div className={s.error}>{error}</div>}
            <div className={s.hint}>Base + each feature, and base + each combination, train on the full
              splits and are scored on test. Progress and results open on the Results page.</div>
            <button className={s.primary} disabled={!picked.length || starting} onClick={run}>
              {starting ? 'Starting…' : `Evaluate ${picked.length} feature${picked.length === 1 ? '' : 's'}`
                + (Object.keys(combos).length ? ` + ${Object.keys(combos).length} combination(s)` : '')}
            </button>
          </section>

        </div>
      </div>
    </div>
  )
}

/** Every word of the query, case aside, found somewhere in the texts. */
export function matches(query: string, texts: (string | null | undefined)[]): boolean {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (!words.length) return true
  const blob = texts.filter(Boolean).join(' ').toLowerCase()
  return words.every((w) => blob.includes(w))
}

/** The list's search: filters as you type; Esc clears it. */
export function SearchBox({ value, onChange, shown, total, placeholder }: {
  value: string; onChange: (v: string) => void; shown: number; total: number; placeholder: string
}) {
  return (
    <div className={s.searchRow}>
      <input className={s.search} type="search" value={value} placeholder={placeholder}
             aria-label="Search the list"
             onChange={(e) => onChange(e.target.value)}
             onKeyDown={(e) => { if (e.key === 'Escape') onChange('') }} />
      {value.trim() && <span className={s.meta}>{shown} of {total}</span>}
    </div>
  )
}

function PoolRow({ feature: f, picked, open, onPick, onOpen, onRemove }: {
  feature: PoolFeature; picked: boolean; open: boolean
  onPick: () => void; onOpen: () => void; onRemove: () => Promise<void>
}) {
  const blocked = f.missing_linkage.length > 0
  return (
    <div className={`${s.feature} ${picked ? s.on : ''} ${open ? s.featureOpen : ''}`}>
      <div className={s.featureRow}>
        <input type="checkbox" checked={picked} onChange={onPick} disabled={blocked}
               title={blocked ? `No linkage recorded for ${f.missing_linkage.join(', ')}` : 'select for evaluation'} />
        <button className={s.featureBody} onClick={onOpen}>
          <span className={s.featureTop}>
            <span className={s.chev}>{open ? '▾' : '▸'}</span>
            <span className={s.name}>{f.name}</span>
            <span className={s.level}>{f.level}</span>
            <span className={s.delta} title="screen gain">Δ {signed(f.delta)}</span>
          </span>
          <span className={s.desc}>{f.description}</span>
          {f.sources.length > 0 && <span className={s.sources}>{f.sources.join(' + ')}</span>}
          {blocked && <span className={s.missing}>no linkage recorded for {f.missing_linkage.join(', ')}</span>}
        </button>
      </div>
      {open && (
        <div className={s.disclosure}>
          <div className={s.disclosureHead}>
            <span>feature script · {f.key}</span>
            <ConfirmButton label="Remove from pool" confirm="Remove this intent?" onConfirm={onRemove} />
          </div>
          <pre className={s.code}>{f.code}</pre>
          {Object.entries(f.linkage).map(([source, code]) => (
            <details key={source} className={s.linkage}>
              <summary>linkage used for {source}</summary>
              <pre className={s.code}>{code}</pre>
            </details>
          ))}
        </div>
      )}
    </div>
  )
}
