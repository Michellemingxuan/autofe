import { useEffect, useMemo, useRef, useState } from 'react'
import type { BoardRow } from '../types'
import s from './BoardFilters.module.css'

/* Notion-style filters for the logging board: the user adds a filter per
 * property, each a chip that reads as a sentence ("Status is feature – failed")
 * and opens a small editor. Filters combine with AND; one with nothing chosen
 * yet does not filter. */

type Kind = 'select' | 'number' | 'text'

type Property = {
  key: string
  label: string
  kind: Kind
  value: (r: BoardRow) => string | number | null     // what a row has for it
  hint?: string
}

export type Rule = {
  id: number
  prop: string
  values: string[]                 // select: any of these
  op: '≥' | '≤'                    // number
  number: string                   // number: the threshold, as typed
  text: string                     // text: contains
}

const ENGINE: Record<string, string> = {
  mechanism: 'Mechanism', contrastive: 'Contrastive', analogy: 'Analogy',
  morphological: 'Morphological', others: 'Others',
}

// The statuses each type of row can have, in the order they are offered.
export const TYPE_STATUSES: Record<string, string[]> = {
  feature: ['verified', 'failed'],
  request: ['kept', 'dropped', 'pending'],
  idea: ['not tried'],
}

export const PROPERTIES: Property[] = [
  { key: 'direction', label: 'Direction', kind: 'select', value: (r) => r.direction || 'Open exploration' },
  { key: 'type', label: 'Type', kind: 'select', value: (r) => r.kind,
    hint: 'feature, request (L3 data), or idea nothing was built from yet' },
  { key: 'status', label: 'Status', kind: 'select', value: (r) => `${r.kind} – ${r.status}` },
  { key: 'engine', label: 'Engine', kind: 'select', value: (r) => (r.engine ? ENGINE[r.engine] ?? r.engine : 'not recorded') },
  { key: 'idea_status', label: 'Idea status', kind: 'select', value: (r) => r.idea_status || null },
  { key: 'level', label: 'Level', kind: 'select', value: (r) => r.level || null },
  { key: 'evaluation', label: 'C4 (evaluation)', kind: 'select', value: (r) => r.sent || null,
    hint: 'verified features only' },
  { key: 'max_model_rho', label: 'C1 (redundancy) max |ρ|', kind: 'number', value: (r) => r.max_model_rho },
  { key: 'variable_gini', label: 'C2 (variable Gini)', kind: 'number',
    value: (r) => r.variable_gini ?? null },
  { key: 'gini_gain', label: 'C3 (screen gain)', kind: 'number', value: (r) => r.gini_gain },
  { key: 'p_useful', label: 'P(useful)', kind: 'number', value: (r) => r.p_useful },
  { key: 'idea', label: 'Idea', kind: 'text', value: (r) => `${r.idea} ${r.idea_description}` },
  { key: 'feature', label: 'Feature', kind: 'text', value: (r) => `${r.name} ${r.description}` },
]
const BY_KEY = Object.fromEntries(PROPERTIES.map((p) => [p.key, p]))

/** Whether a row passes every filter. */
export function applies(rules: Rule[], r: BoardRow): boolean {
  return rules.every((rule) => {
    const prop = BY_KEY[rule.prop]
    const v = prop.value(r)
    if (prop.kind === 'select') return !rule.values.length || (v !== null && rule.values.includes(String(v)))
    if (prop.kind === 'text') {
      return !rule.text.trim() || String(v ?? '').toLowerCase().includes(rule.text.trim().toLowerCase())
    }
    const limit = parseFloat(rule.number)
    if (Number.isNaN(limit)) return true
    if (v === null || v === undefined) return false
    return rule.op === '≥' ? Number(v) >= limit : Number(v) <= limit
  })
}

/** The choices a select filter offers: what the rows have - and for Status, only
 *  the statuses of the types the Type filter keeps. */
function choices(prop: Property, rows: BoardRow[], rules: Rule[]): string[] {
  if (prop.key === 'type') return Object.keys(TYPE_STATUSES)
  if (prop.key === 'status') {
    const types = rules.find((r) => r.prop === 'type' && r.values.length)?.values ?? Object.keys(TYPE_STATUSES)
    return types.flatMap((t) => TYPE_STATUSES[t].map((st) => `${t} – ${st}`))
  }
  if (prop.key === 'evaluation') return ['not evaluated', 'evaluated', 'evaluated · removed', 'removed']
  if (prop.key === 'idea_status') return ['explore', 'active', 'low']
  const seen = new Set<string>()
  for (const r of rows) {
    const v = prop.value(r)
    if (v !== null && v !== '') seen.add(String(v))
  }
  return [...seen].sort()
}

/** A rule as the chip reads it. */
function sentence(rule: Rule): { what: string; set: boolean } {
  const prop = BY_KEY[rule.prop]
  if (prop.kind === 'select') {
    return rule.values.length
      ? { what: `is ${rule.values.length > 2 ? `${rule.values.slice(0, 2).join(', ')} +${rule.values.length - 2}` : rule.values.join(', ')}`, set: true }
      : { what: '', set: false }
  }
  if (prop.kind === 'text') return rule.text.trim() ? { what: `contains "${rule.text.trim()}"`, set: true } : { what: '', set: false }
  return rule.number.trim() ? { what: `${rule.op} ${rule.number.trim()}`, set: true } : { what: '', set: false }
}

type Props = { rows: BoardRow[]; rules: Rule[]; onChange: (rules: Rule[]) => void }

export function BoardFilters({ rows, rules, onChange }: Props) {
  const [adding, setAdding] = useState(false)
  const [editing, setEditing] = useState<number | null>(null)
  const ref = useRef<HTMLDivElement>(null)
  const nextId = useRef(1)

  // A click anywhere else closes the menu or the editor.
  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) { setAdding(false); setEditing(null) }
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const update = (id: number, patch: Partial<Rule>) => {
    let next = rules.map((r) => (r.id === id ? { ...r, ...patch } : r))
    // A status of a type the Type filter no longer keeps goes with it.
    const types = next.find((r) => r.prop === 'type' && r.values.length)?.values
    if (types) {
      next = next.map((r) => (r.prop === 'status'
        ? { ...r, values: r.values.filter((v) => types.includes(v.split(' – ')[0])) } : r))
    }
    onChange(next)
  }
  const add = (prop: string) => {
    const id = nextId.current++
    onChange([...rules, { id, prop, values: [], op: '≥', number: '', text: '' }])
    setAdding(false)
    setEditing(id)
  }
  const remove = (id: number) => { onChange(rules.filter((r) => r.id !== id)); setEditing(null) }
  const unused = useMemo(() => PROPERTIES.filter((p) => !rules.some((r) => r.prop === p.key)), [rules])

  return (
    <div className={s.bar} ref={ref}>
      {rules.map((rule) => {
        const prop = BY_KEY[rule.prop]
        const { what, set } = sentence(rule)
        return (
          <div key={rule.id} className={s.wrap}>
            <button className={`${s.chip} ${set ? s.chipSet : ''}`}
                    onClick={() => { setEditing(editing === rule.id ? null : rule.id); setAdding(false) }}>
              <span className={s.prop}>{prop.label}</span>
              {set ? <span className={s.what}>{what}</span> : <span className={s.empty}>any</span>}
              <span className={s.caret}>▾</span>
            </button>
            {editing === rule.id && (
              <div className={s.pop}>
                <div className={s.popHead}>
                  <span>{prop.label}{prop.hint ? <span className={s.hint}> · {prop.hint}</span> : null}</span>
                  <button className={s.remove} onClick={() => remove(rule.id)}>Remove filter</button>
                </div>
                {prop.kind === 'select' && (
                  <div className={s.options}>
                    {choices(prop, rows, rules).map((c) => (
                      <label key={c} className={s.option}>
                        <input type="checkbox" checked={rule.values.includes(c)}
                               onChange={(e) => update(rule.id, {
                                 values: e.target.checked ? [...rule.values, c] : rule.values.filter((v) => v !== c),
                               })} />
                        <span>{c}</span>
                      </label>
                    ))}
                  </div>
                )}
                {prop.kind === 'number' && (
                  <div className={s.numberRow}>
                    <select value={rule.op} onChange={(e) => update(rule.id, { op: e.target.value as Rule['op'] })}>
                      <option value="≥">≥</option>
                      <option value="≤">≤</option>
                    </select>
                    <input autoFocus type="number" step="any" value={rule.number} placeholder="value"
                           onChange={(e) => update(rule.id, { number: e.target.value })} />
                  </div>
                )}
                {prop.kind === 'text' && (
                  <input autoFocus className={s.textInput} value={rule.text} placeholder="contains…"
                         onChange={(e) => update(rule.id, { text: e.target.value })} />
                )}
              </div>
            )}
          </div>
        )
      })}

      <div className={s.wrap}>
        <button className={s.add} disabled={!unused.length}
                onClick={() => { setAdding((v) => !v); setEditing(null) }}>+ Filter</button>
        {adding && (
          <div className={s.pop}>
            {unused.map((p) => (
              <button key={p.key} className={s.menuItem} onClick={() => add(p.key)}>
                <span>{p.label}</span>
                <span className={s.kindTag}>{p.kind === 'number' ? '≥ ≤' : p.kind === 'text' ? 'contains' : 'is'}</span>
              </button>
            ))}
          </div>
        )}
      </div>
      {rules.length > 0 && (
        <button className={s.clear} onClick={() => { onChange([]); setEditing(null) }}>Clear</button>
      )}
    </div>
  )
}
