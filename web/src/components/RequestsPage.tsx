import { useMemo, useState } from 'react'
import { api, usePolled } from '../api'
import { BEYOND, type RequestItem } from '../types'
import { ConfirmButton } from './ConfirmButton'
import { matches, SearchBox } from './EvaluatePage'
import s from './EvaluatePage.module.css'
import r from './RequestsPage.module.css'

// A filter: every request, one scope's by its keyword, or those beyond scope.
type Filter = 'all' | string

const scopeLabel = (scope: string) => (scope === BEYOND ? 'beyond scope' : `${scope} · SQL`)

/**
 * Step 3, the request set. Every data request a direction kept - its challenge
 * found the data missing - from every run, in one place: what to take to the
 * data owners. Within a scope (CAS, say) a request carries validated SQL; beyond
 * scope, a description of the data it needs.
 */
export function RequestsPage() {
  const [items, refresh] = usePolled(api.requests, [], 10000)
  const [filter, setFilter] = useState<Filter>('all')
  const [opened, setOpened] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [query, setQuery] = useState('')

  const shown = useMemo(() => (items ?? []).filter((x) =>
    (filter === 'all' || x.scope === filter)
    && matches(query, [x.source_name, x.intent, x.gap, x.features, x.data, x.sql, x.scope, x.direction,
                       x.run_id, x.challenge?.reasoning, ...x.tables, ...(x.columns ?? [])])),
  [items, filter, query])
  const groups = useMemo(() => {
    const out: { run_id: string; direction: string; items: RequestItem[] }[] = []
    for (const x of shown) {
      let g = out.find((y) => y.run_id === x.run_id)
      if (!g) out.push(g = { run_id: x.run_id, direction: x.direction, items: [] })
      g.items.push(x)
    }
    return out
  }, [shown])
  const all = items ?? []
  const inScope = all.filter((x) => x.kind === 'in_scope')
  // The scopes present, then beyond scope - the filter row.
  const scopes = [...new Set(inScope.map((x) => x.scope))].sort()
  const filters: Filter[] = ['all', ...scopes, ...(all.length > inScope.length ? [BEYOND] : [])]
  const sql = inScope.map((x) => `-- ${x.intent} ${x.source_name} · ${x.scope} (run ${x.run_id})\n${x.sql.trim()}`).join('\n\n')

  return (
    <div className={s.page}>
      <header className={s.header}>
        <span className={`eyebrow ${s.eyebrow}`}>Step 3 · Evaluate</span>
        <h1 className={s.title}>Request set</h1>
      </header>

      <div className={s.body}>
        <section className={s.pool}>
          <div className={s.panelHead}>
            <span className={`eyebrow ${s.panelTitle}`}>Kept data requests</span>
            <div className={r.filters}>
              {filters.map((k) => (
                <button key={k} className={`${r.filter} ${filter === k ? r.filterOn : ''}`}
                        onClick={() => setFilter(k)}>
                  {k === 'all' ? 'All' : k === BEYOND ? 'Beyond scope' : k}{' '}
                  {k === 'all' ? all.length : all.filter((x) => x.scope === k).length}
                </button>
              ))}
            </div>
          </div>
          <SearchBox value={query} onChange={setQuery} shown={shown.length} total={all.length}
                     placeholder="Search name, rationale, features, columns, tables or SQL" />
          <div className={s.scroll}>
            {items && all.length === 0 && (
              <div className={s.empty}>No kept data requests yet - run a direction with L3 in Discover.</div>
            )}
            {items && all.length > 0 && shown.length === 0 && (
              <div className={s.empty}>{query.trim() ? `No request matches “${query.trim()}”.` : 'None of this kind.'}</div>
            )}
            {groups.map((g) => (
              <div key={g.run_id} className={s.group}>
                <div className={s.groupHead}>
                  <span className={s.groupDirection}>{g.direction}</span>
                  <span className={s.meta}>{g.run_id}</span>
                </div>
                {g.items.map((x) => (
                  <RequestRow key={x.key} item={x} open={opened === x.key}
                              onOpen={() => setOpened(opened === x.key ? null : x.key)}
                              onRemove={async () => { await api.deleteRequest(x.run_id, x.intent); refresh() }} />
                ))}
              </div>
            ))}
          </div>
        </section>

        <div className={s.side}>
          <section className={s.card}>
            <div className={`eyebrow ${s.panelTitle}`}>For the data owners</div>
            <div className={r.counts}>
              <span><b>{inScope.length}</b> within a scope{scopes.length ? ` (${scopes.join(', ')})` : ''}, with validated SQL</span>
              <span><b>{all.length - inScope.length}</b> beyond scope, each with the data it needs</span>
              <span><b>{new Set(all.map((x) => x.run_id)).size}</b> direction{new Set(all.map((x) => x.run_id)).size === 1 ? '' : 's'}</span>
            </div>
            <div className={s.hint}>One document with every kept request: why it is needed, the
              features it would enable, the challenge, and its SQL or the data it needs.</div>
            <a className={`${s.primary} ${r.download} ${all.length ? '' : r.disabled}`}
               href={all.length ? api.requestSetUrl : undefined} download="data_requests.md">
              Download all (.md)
            </a>
            <button className={s.secondary} disabled={!inScope.length}
                    onClick={() => { navigator.clipboard?.writeText(sql); setCopied(true) }}>
              {copied ? 'Copied' : `Copy all SQL (${inScope.length})`}
            </button>
            <div className={s.hint}>Removing a request takes it off this set; later directions may then
              ask for the same data again.</div>
          </section>
        </div>
      </div>
    </div>
  )
}

function RequestRow({ item: x, open, onOpen, onRemove }: {
  item: RequestItem; open: boolean; onOpen: () => void; onRemove: () => Promise<void>
}) {
  const [copied, setCopied] = useState(false)
  const features = x.features.split('\n').map((f) => f.replace(/^[-*\s]+/, '').trim()).filter(Boolean)
  const c = x.challenge
  return (
    <div className={`${s.feature} ${open ? s.featureOpen : ''}`}>
      <div className={s.featureRow}>
        <button className={s.featureBody} onClick={onOpen}>
          <span className={s.featureTop}>
            <span className={s.chev}>{open ? '▾' : '▸'}</span>
            <span className={s.name}>{x.source_name}</span>
            <span className={`${r.kind} ${r[x.kind]}`}>{scopeLabel(x.scope)}</span>
            {c && <span className={r.verdict} title="the challenge: can current data supply it?">{c.verdict}</span>}
          </span>
          <span className={`${s.desc} ${open ? '' : r.clamp}`}>{x.gap}</span>
        </button>
      </div>
      {open && (
        <div className={s.disclosure}>
          <div className={s.disclosureHead}>
            <span>{x.intent} · run {x.run_id}</span>
            <ConfirmButton label="Remove from set" confirm="Remove this request?" onConfirm={onRemove} />
          </div>
          {features.length > 0 && (
            <div className={r.field}>
              <span className={r.label}>Features it would enable</span>
              <div className={r.chips}>{features.map((f) => <code key={f}>{f}</code>)}</div>
            </div>
          )}
          {x.sql ? (
            <div className={r.field}>
              <span className={r.label}>
                Reads ({x.scope}) {x.tables.map((t) => <code key={t}>{t}</code>)}
                <button className={s.link} onClick={() => { navigator.clipboard?.writeText(x.sql); setCopied(true) }}>
                  {copied ? 'copied' : 'copy SQL'}</button>
              </span>
              <pre className={s.code}>{x.sql}</pre>
            </div>
          ) : (
            <div className={r.field}>
              <span className={r.label}>The data it needs</span>
              <span className={r.text}>{x.data}</span>
            </div>
          )}
          {c && (
            <div className={r.field}>
              <span className={r.label}>Challenge · can current data supply it? {c.verdict}</span>
              <span className={r.text}>{c.reasoning}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
