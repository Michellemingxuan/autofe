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
/** A combination being built: its members, and the name it is evaluated under. */
type Combo = { id: string; name: string; keys: string[] }

// What a dragged feature carries: its pool key.
const DRAG = 'application/x-autofe-feature'
const dragged = (e: React.DragEvent) => e.dataTransfer.types.includes(DRAG)
const startDrag = (key: string) => (e: React.DragEvent) => {
  e.dataTransfer.setData(DRAG, key)
  e.dataTransfer.effectAllowed = 'copy'
}
const slug = (name: string) => name.trim().replace(/\W+/g, '_').replace(/^_+|_+$/g, '')

export function EvaluatePage({ onStarted }: { onStarted: (id: string) => void }) {
  const [pool, refreshPool] = usePolled(api.features, [], 10000)
  const [picked, setPicked] = useState<string[]>([])           // evaluated alone
  const [combos, setCombos] = useState<Combo[]>([])
  // Then base + the singles together: every one that reached modelling, or the passing ones.
  const [together, setTogether] = useState<'all' | 'passing' | 'off'>('all')
  const [over, setOver] = useState<string | null>(null)        // the drop zone under the drag
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
    setCombos((cs) => cs.map((c) => ({ ...c, keys: c.keys.filter((k) => alive.has(k)) }))
      .filter((c) => c.keys.length))
  }, [pool])

  const usable = (key: string) => !!byKey[key] && !byKey[key].missing_linkage.length
  const toggle = (key: string) =>
    setPicked((p) => (p.includes(key) ? p.filter((k) => k !== key) : [...p, key]))
  /** Into a combination - a new one when `id` is null. */
  const addTo = (id: string | null, key: string) => {
    if (!usable(key)) return
    setCombos((cs) => {
      if (id === null) {
        let n = cs.length + 1
        while (cs.some((c) => c.name === `combo_${n}`)) n++
        return [...cs, { id: `c${Date.now()}`, name: `combo_${n}`, keys: [key] }]
      }
      return cs.map((c) => (c.id === id && !c.keys.includes(key) ? { ...c, keys: [...c.keys, key] } : c))
    })
  }
  const takeOut = (id: string, key: string) =>
    setCombos((cs) => cs.map((c) => (c.id === id ? { ...c, keys: c.keys.filter((k) => k !== key) } : c))
      .filter((c) => c.keys.length))
  const rename = (id: string, name: string) => setCombos((cs) => cs.map((c) => (c.id === id ? { ...c, name } : c)))

  /** Drop-zone props: highlighted while a feature is over it. */
  const zone = (id: string, onDrop: (key: string) => void) => ({
    onDragOver: (e: React.DragEvent) => { if (dragged(e)) { e.preventDefault(); setOver(id) } },
    onDragLeave: () => setOver((o) => (o === id ? null : o)),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault()
      setOver(null)
      const key = e.dataTransfer.getData(DRAG)
      if (key) onDrop(key)
    },
  })

  // A combination needs two features; names are made unique as they are sent.
  const ready = combos.filter((c) => c.keys.length >= 2)
  const named = (() => {
    const out: Record<string, string[]> = {}
    ready.forEach((c, i) => {
      let name = slug(c.name) || `combo_${i + 1}`
      while (out[name]) name += '_2'
      out[name] = c.keys
    })
    return out
  })()
  const withTogether = together !== 'off' && picked.length >= 2
  const run = async () => {
    setError(null)
    setStarting(true)
    try {
      const { eval_id } = await api.evaluate(picked, named, withTogether ? together : null)
      onStarted(eval_id)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setStarting(false)
    }
  }
  const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`
  const label = [picked.length ? plural(picked.length, 'feature') : '',
                 ready.length ? plural(ready.length, 'combination') : ''].filter(Boolean).join(' + ')

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
            <span className={s.meta}>{pool?.length ?? 0} verified · {groups.length} directions · tick, or drag into a combination</span>
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
            <ol className={s.howto}>
              <li><b>Alone</b> - tick a feature in the pool: base + it, a model of its own.</li>
              <li><b>Together</b> - drag features from the pool onto <i>New combination</i>: base + all of them,
                one model. Drop more onto a combination to add them, and name it if you like.</li>
              <li>Then one more model with the singles together - all of them, or only those that pass.</li>
            </ol>

            <div className={`${s.zone} ${over === 'alone' ? s.zoneOver : ''}`}
                 {...zone('alone', (k) => { if (usable(k) && !picked.includes(k)) setPicked((p) => [...p, k]) })}>
              <div className={s.zoneHead}>
                <span className={s.zoneTitle}>Alone</span>
                <span className={s.meta}>{picked.length ? `${picked.length} · base + each` : 'tick features in the pool'}</span>
              </div>
              {picked.length > 0 && (
                <div className={s.chips}>
                  {picked.map((k) => (
                    <span key={k} className={s.chip} draggable onDragStart={startDrag(k)}
                          title="Drag onto a combination to evaluate it there too">
                      <span className={s.grip} aria-hidden>⠿</span>{byKey[k]?.name ?? k}
                      <button className={s.chipX} title="Not alone" onClick={() => toggle(k)}>×</button>
                    </span>
                  ))}
                </div>
              )}
            </div>

            {combos.map((c) => (
              <div key={c.id} className={`${s.zone} ${s.comboZone} ${over === c.id ? s.zoneOver : ''}`}
                   {...zone(c.id, (k) => addTo(c.id, k))}>
                <div className={s.zoneHead}>
                  <span className={s.comboName}>⊕</span>
                  <input className={s.comboInput} value={c.name} aria-label="Combination name"
                         onChange={(e) => rename(c.id, e.target.value)} />
                  <button className={s.remove} title="Remove this combination"
                          onClick={() => setCombos((cs) => cs.filter((x) => x.id !== c.id))}>×</button>
                </div>
                <div className={s.chips}>
                  <span className={s.basePill}>base +</span>
                  {c.keys.map((k) => (
                    <span key={k} className={`${s.chip} ${s.chipOn}`} draggable onDragStart={startDrag(k)}>
                      {byKey[k]?.name ?? k}
                      <button className={s.chipX} title="Take it out" onClick={() => takeOut(c.id, k)}>×</button>
                    </span>
                  ))}
                  <select className={s.addSelect} value="" aria-label="Add a feature"
                          onChange={(e) => { if (e.target.value) addTo(c.id, e.target.value) }}>
                    <option value="">+ add…</option>
                    {(pool ?? []).filter((f) => !c.keys.includes(f.key) && usable(f.key)).map((f) => (
                      <option key={f.key} value={f.key}>{f.name}</option>
                    ))}
                  </select>
                </div>
                {c.keys.length < 2 && <div className={s.hint}>Add one more feature - a combination needs two.</div>}
              </div>
            ))}

            <div className={`${s.zone} ${s.newZone} ${over === 'new' ? s.zoneOver : ''}`}
                 {...zone('new', (k) => addTo(null, k))}>
              <span>⊕ New combination</span>
              <span className={s.meta}>drag a feature here</span>
            </div>

            <div className={`${s.together} ${picked.length < 2 ? s.togetherOff : ''}`}>
              <span>Then, one more model:</span>
              <select className={s.togetherSelect} value={together} disabled={picked.length < 2}
                      onChange={(e) => setTogether(e.target.value as typeof together)}>
                <option value="all">base + all new features</option>
                <option value="passing">base + only the features that pass</option>
                <option value="off">none</option>
              </select>
              <span className={s.hint}>
                {picked.length < 2 ? 'needs two features alone'
                  : together === 'all' ? 'every feature evaluated alone, together - trained after them; it shows under Combinations'
                  : together === 'passing' ? 'trained after the verdicts, if two or more pass; it shows under Combinations'
                  : ''}
              </span>
            </div>

            {error && <div className={s.error}>{error}</div>}
            <div className={s.hint}>Every model trains on the full splits and is scored on test; progress and
              results open on Metrics results, the target cases on Inference results.</div>
            <button className={s.primary} disabled={(!picked.length && !ready.length) || starting} onClick={run}>
              {starting ? 'Starting…' : label ? `Evaluate ${label}${withTogether ? (together === 'all' ? ' + all together' : ' + the passing together') : ''}` : 'Evaluate'}
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
    <div className={`${s.feature} ${picked ? s.on : ''} ${open ? s.featureOpen : ''}`}
         draggable={!blocked} onDragStart={startDrag(f.key)}>
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
          <Evaluated status={f.evaluation} />
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

// The combinations an evaluation adds itself, by what they hold.
const AUTO_COMBO: Record<string, string> = { all_new: 'all new features', all_passing: 'every passing feature' }

/** What evaluations made of a pool feature: alone - its verdict, or why it was not
 *  trained - and the combinations it was in. */
function Evaluated({ status }: { status?: PoolFeature['evaluation'] }) {
  if (!status) return <span className={s.notYet}>not evaluated yet</span>
  const one = status.single
  return (
    <span className={s.evalLine}>
      {one && (one.trained === false
        ? <span className={`${s.evalTag} ${s.evalDropped}`} title={one.not_trained ?? undefined}>
            {one.not_trained?.startsWith('training failed') ? 'training failed' : 'dropped before modelling'}</span>
        : <span className={`${s.evalTag} ${one.verdict === 'PASS' ? s.evalPass : s.evalFail}`}
                title={one.reason || undefined}>
            evaluated · {one.verdict ?? '–'}{one.gini_gain != null ? ` · test Δ ${signed(one.gini_gain)}` : ''}</span>)}
      {status.combinations.length > 0 && (
        <span className={s.evalCombo}>in {status.combinations.map((c) =>
          `⊕ ${AUTO_COMBO[c.name] ?? c.name} (${c.members.join(' + ')})`).join(', ')}</span>
      )}
    </span>
  )
}
