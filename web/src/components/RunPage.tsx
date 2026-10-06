import { useEffect, useMemo, useState } from 'react'
import { api, rerunDraft, streamUrl, useEventStream } from '../api'
import { deriveRun } from '../derive'
import { Trace } from './Trace'
import { Timeline } from './Timeline'
import { ConfirmButton } from './ConfirmButton'
import s from './RunPage.module.css'

const STATUS_LABEL: Record<string, string> = {
  idle: 'loading', running: 'in progress', waiting: 'waiting for you', done: 'finished', error: 'error',
}

/** One direction: what the agent is doing (trace) beside where it is (timeline). */
export function RunPage({ runId, onChange, onDeleted, onRerun }: {
  runId: string; onChange: () => void; onDeleted: () => void
  onRerun: () => void                  // opens the direction form, filled in
}) {
  const { events, connected, gone } = useEventStream(streamUrl.run(runId))
  const view = useMemo(() => deriveRun(events), [events])
  const [jumpTo, setJumpTo] = useState<{ id: number; n: number } | null>(null)
  const [inView, setInView] = useState<number | null>(null)

  // The rail's counts change when a run finishes or verifies a feature.
  const verified = view.ledger.filter((r) => r.verified && !r.deleted).length
  useEffect(onChange, [view.status, verified])

  const live = view.status === 'running' || view.status === 'waiting'
  if (gone && !events.length) {
    return (
      <div className={s.page}>
        <div className={s.gone}>This run no longer exists - it may have been deleted. Pick another
          direction on the left, or start a new one.</div>
      </div>
    )
  }
  const l3 = view.params?.levels?.length === 1 && view.params.levels[0] === 'L3'
  const p = view.params
  const kept = view.requests.filter((r) => r.status === 'kept').length

  return (
    <div className={s.page}>
      <header className={s.header}>
        <div className={s.headText}>
          <span className={`eyebrow ${s.eyebrow}`}>Discover · direction</span>
          <div className={s.titleRow}>
            <h1 className={s.title}>{runId}</h1>
            <span className={`${s.pill} ${s[view.status]}`}>
              <span className={s.pillDot} />{STATUS_LABEL[view.status]}
            </span>
            {!connected && live && <span className={s.offline}>reconnecting…</span>}
          </div>
          {/* The direction can be a paragraph: up to two lines, the rest on hover. */}
          <div className={s.direction} title={view.direction}>{view.direction || 'Loading…'}</div>
          {p && (
            <div className={s.params}>
              <span>{p.model}</span><span>{p.engine}</span>
              <span title={view.quota && Object.keys(view.quota).length > 1
                ? 'Intents split by level at random, weighted L2 > L1 > L3' : undefined}>
                {view.quota && Object.keys(view.quota).length > 1
                  ? Object.entries(view.quota).map(([lv, n]) => `${lv} ×${n}`).join(' · ')
                  : p.levels.join(' · ')}</span>
              {!l3 && <span>Gini Δ &gt; {p.min_gini_gain}</span>}
              {!l3 && p.min_capture_gain != null && <span>capture Δ &gt; {p.min_capture_gain}</span>}
              <span>{p.sources.length ? p.sources.join(', ') : 'no sources'}</span>
            </div>
          )}
        </div>
        <div className={s.headRight}>
          <div className={s.counter} title={`of ${view.K || '–'} intents`}>
            <span className={s.num}>{l3 ? view.requests.length : view.ledger.length}</span>
            <span className={s.of}>proposed</span>
          </div>
          <div className={s.counter}>
            <span className={`${s.num} ${s.good}`}>{l3 ? kept : verified}</span>
            <span className={s.of}>{l3 ? 'kept' : 'verified'}</span>
          </div>
          <div className={s.actions}>
            {live && <button className={s.stop} onClick={() => api.cancel('runs', runId)}>Stop</button>}
            {l3 && view.requests.length > 0 && (
              <a className={s.action} href={api.requestsUrl(runId)} download
                 title="The kept requests' SQL and rationale, as markdown">Download</a>
            )}
            {!live && view.status !== 'idle' && view.params && (
              <button className={s.action} title="Open the direction form, filled in - the new run replaces this one"
                      onClick={() => {
                        rerunDraft.set({ direction: view.direction, params: view.params!, replaces: runId })
                        onRerun()
                      }}>Re-run</button>
            )}
            {!live && view.status !== 'idle' && (
              <ConfirmButton label="Delete" confirm="Delete this direction and its intents?"
                             onConfirm={async () => { await api.deleteRun(runId); onDeleted() }} />
            )}
          </div>
        </div>
      </header>

      <div className={s.body}>
        <Trace jobKind="runs" jobId={runId} view={view} jumpTo={jumpTo} onInView={setInView}
               onDeleteIntent={async (name) => { await api.deleteIntent(runId, name); onChange() }} />
        <Timeline view={view} inView={inView}
                  onSelect={(id) => setJumpTo({ id, n: Date.now() })} />
      </div>
    </div>
  )
}
