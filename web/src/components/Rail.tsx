import { navigate, type Page, type Route } from '../App'
import type { EvalSummary, RunSummary, SourceState, Workspace } from '../types'
import s from './Rail.module.css'

type Props = {
  route: Route
  workspace: Workspace | null
  runs: RunSummary[]
  evaluations: EvalSummary[]
}

export const SOURCE_STATE: Record<SourceState, { label: string; hint: string }> = {
  linked: { label: 'linked', hint: 'Linkage confirmed - features can use it' },
  needs_linkage: { label: 'needs linkage', hint: 'Data is here; the agent will propose a linkage for you to confirm' },
  schema_only: { label: 'schema only', hint: 'Only the sample JSON is here; drop the data file to use it' },
}

const STEPS: { page: Page; name: string;
                hint: (w: Workspace | null, r: RunSummary[], e: EvalSummary[]) => string }[] = [
  { page: 'setup', name: 'Setup',
    hint: (w) => w ? `${w.sources.filter((x) => x.state === 'linked').length}/${w.sources.length} sources linked` : '' },
  { page: 'discover', name: 'Discover',
    hint: (_, r) => `${r.length} direction${r.length === 1 ? '' : 's'} · ${r.reduce((n, x) => n + x.verified, 0)} verified` },
  { page: 'evaluate', name: 'Evaluate',
    hint: (_, __, e) => `${e.length} evaluation${e.length === 1 ? '' : 's'}` },
]

// The pages under step 3.
const EVALUATE: Page[] = ['evaluate', 'requests', 'results']

const when = (ts: number) => new Date(ts * 1000).toLocaleString([], {
  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
})

/** The dark rail: the three steps, and what each has made so far. */
export function Rail({ route, workspace, runs, evaluations }: Props) {
  return (
    <aside className={s.rail}>
      <div className={s.brand} title={workspace?.name ? `workspace: ${workspace.name}` : undefined}>
        <span className={s.logo}>af</span>
        <div>
          <div className={s.brandName}>AutoFE</div>
          <div className={s.brandSub}>Model agent</div>
        </div>
      </div>

      <nav className={s.steps}>
        {STEPS.map((step, i) => (
          <button key={step.page}
                  className={`${s.step} ${route.page === step.page
                    || (step.page === 'evaluate' && EVALUATE.includes(route.page)) ? s.stepOn : ''}`}
                  onClick={() => navigate(step.page)}>
            <span className={s.stepNum}>{i + 1}</span>
            <span className={s.stepText}>
              <span className={s.stepName}>{step.name}</span>
              <span className={s.stepHint}>{step.hint(workspace, runs, evaluations)}</span>
            </span>
          </button>
        ))}
      </nav>

      {route.page === 'setup' && (
        <div className={s.section}>
          <div className={`eyebrow ${s.sectionHead}`}>Sources</div>
          {!workspace?.sources.length && <div className={s.empty}>No additional data yet</div>}
          {workspace?.sources.map((src) => (
            <div key={src.name} className={s.source} title={SOURCE_STATE[src.state].hint}>
              <span className={s.sourceName}>{src.name}</span>
              <span className={`${s.badge} ${s[src.state]}`}>{SOURCE_STATE[src.state].label}</span>
            </div>
          ))}
        </div>
      )}

      {route.page === 'discover' && (
        <>
          <button className={`${s.newBtn} ${!route.id ? s.newOn : ''}`}
                  onClick={() => navigate('discover')}>+ New direction</button>
          <div className={s.section}>
            <div className={`eyebrow ${s.sectionHead}`}>Directions</div>
            {runs.length === 0 && <div className={s.empty}>None yet</div>}
            {runs.map((r) => (
              <button key={r.run_id}
                      className={`${s.item} ${r.run_id === route.id ? s.active : ''}`}
                      onClick={() => navigate('discover', r.run_id)} title={r.direction}>
                <span className={`${s.dot} ${s[r.status]}`} />
                <span className={s.itemText}>
                  <span className={s.itemTitle}>{r.direction}</span>
                  <span className={s.itemMeta}>{when(r.started)} · {r.mode === 'l3'
                    ? `${r.requests} data request${r.requests === 1 ? '' : 's'}` : `${r.verified} verified`}</span>
                </span>
              </button>
            ))}
          </div>
        </>
      )}

      {EVALUATE.includes(route.page) && (
        <>
          <div className={s.subnav}>
            <button className={`${s.subItem} ${route.page === 'evaluate' ? s.subOn : ''}`}
                    onClick={() => navigate('evaluate')}>Feature set</button>
            <button className={`${s.subItem} ${route.page === 'requests' ? s.subOn : ''}`}
                    onClick={() => navigate('requests')}>Request set</button>
            <button className={`${s.subItem} ${route.page === 'results' ? s.subOn : ''}`}
                    onClick={() => navigate('results')}>Results</button>
          </div>
        </>
      )}
    </aside>
  )
}
