import { useEffect, useState } from 'react'
import type {
  EvalResult, EvalSummary, Ev, Params, PathCheck, PoolFeature, RunSummary, SetupView, Source, Workspace,
} from './types'

// Every backend route the UI uses, in the order of the journey. The server's
// docstring (src/agent/server.py) lists the same contract, and
// tests/test_agent_server.py fails if a path here has no route there.

// The browser says only "Failed to fetch" when no answer came: say what it means.
const NO_ANSWER = 'No answer from the server - it may have stopped, or the request took too '
  + 'long. Check the terminal running agent.server, then try again.'

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch {
    throw new Error(NO_ANSWER)
  }
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.error ?? `${res.status} ${res.statusText}`)
  return body as T
}

const post = <T,>(path: string, body: unknown = {}) =>
  call<T>(path, { method: 'POST', body: JSON.stringify(body) })
const del = <T,>(path: string) => call<T>(path, { method: 'DELETE' })

export type JobKind = 'runs' | 'linkage'

export const api = {
  // 1. Setup
  workspace: () => call<Workspace>('/api/workspace'),
  setup: () => call<SetupView>('/api/setup'),
  applySetup: (values: Record<string, unknown>) => post<SetupView>('/api/setup', { values }),
  resetSetup: () => post<SetupView>('/api/setup/reset'),
  checkPaths: (paths: string[]) => post<PathCheck>('/api/setup/check', { paths }),
  upload: async (kind: string, file: File) => {
    // Multipart, so not through `call`, which sends JSON.
    const form = new FormData()
    form.append('kind', kind)
    form.append('file', file)
    const res = await fetch('/api/uploads', { method: 'POST', body: form })
    const body = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(body.error ?? `${res.status} ${res.statusText}`)
    return body as { path: string; name: string; bytes: number }
  },
  addShotCategory: (category: { name: string; context: string; ids?: string[]; table?: string;
                                 rotate: boolean; batch_size?: number }) =>
    post<{ path: string }>('/api/shots/categories', category),
  generateShots: (shots: number, batches: number) =>
    post<{ path: string }>('/api/shots/clustering', { shots, batches }),
  addSource: (name: string, schema: string, data: string) =>
    post<{ sources: Source[] }>('/api/sources', { name, schema, data }),
  removeSource: (name: string) => del<{ sources: Source[] }>(`/api/sources/${name}`),
  linkageCode: (name: string) => call<{ code: string }>(`/api/sources/${name}/linkage`),
  proposeLinkage: (source: string) => post<{ run_id: string }>('/api/linkage', { source }),

  // Shared by a direction (runs) and a linkage job (linkage)
  approve: (kind: JobKind, id: string, reqId: string, approved: boolean, note: string) =>
    post(`/api/${kind}/${id}/approvals/${reqId}`, { approved, note }),
  cancel: (kind: JobKind, id: string) => post(`/api/${kind}/${id}/cancel`),

  // 2. Discover
  runs: () => call<RunSummary[]>('/api/runs'),
  start: (direction: string, params: Params, replaces?: string) =>
    post<{ run_id: string }>('/api/runs', { direction, params, replaces }),
  deleteRun: (id: string) => del(`/api/runs/${id}`),
  deleteIntent: (runId: string, name: string) => del(`/api/runs/${runId}/intents/${name}`),
  requestsUrl: (runId: string) => `/api/runs/${runId}/requests.md`,

  // 3. Evaluate
  features: () => call<PoolFeature[]>('/api/features'),
  evaluations: () => call<EvalSummary[]>('/api/evaluations'),
  results: () => call<EvalResult[]>('/api/results'),
  clearResults: () => del('/api/results'),
  evaluate: (features: string[], combinations: Record<string, string[]>) =>
    post<{ eval_id: string }>('/api/evaluations', { features, combinations }),
  deleteEvaluation: (id: string) => del(`/api/evaluations/${id}`),
  removeResult: (id: string, variant: string) =>
    del(`/api/evaluations/${id}/variants/${encodeURIComponent(variant)}`),
}

export const streamUrl = {
  run: (id: string) => `/api/runs/${id}/stream`,
  linkage: (id: string) => `/api/linkage/${id}/stream`,
  evaluation: (id: string) => `/api/evaluations/${id}/stream`,
}

const EVENTS = [
  'run_started', 'agent_message', 'tool_started', 'tool_completed', 'skill_loaded',
  'code_status', 'approval_required', 'approval_resolved', 'feature_screened',
  'feature_verified', 'intent_deleted', 'data_request', 'request_challenged',
  'stage_done', 'stage_started', 'agent_nudged', 'source_detected', 'ideas_recorded', 'ideas_sent_back',
  'run_done', 'run_error',
  'eval_started', 'eval_status', 'eval_log', 'eval_done', 'eval_error', 'variant_removed',
]

/**
 * A job's events, live. The server replays its buffer on every (re)connect,
 * so events are kept by `seq` and a reconnect never duplicates or drops one.
 */
export function useEventStream(url: string | null): { events: Ev[]; connected: boolean; gone: boolean } {
  const [events, setEvents] = useState<Ev[]>([])
  const [connected, setConnected] = useState(false)
  // A stream the server refuses (the run was deleted) closes for good; a
  // dropped connection reconnects on its own.
  const [gone, setGone] = useState(false)

  useEffect(() => {
    setEvents([])
    setGone(false)
    if (!url) return
    const seen = new Set<number>()
    let buffer: Ev[] = []
    // A short timer, not requestAnimationFrame: rAF stops in a background
    // tab, which would hold a run's events until the tab is looked at.
    let frame: ReturnType<typeof setTimeout> | 0 = 0
    const flush = () => {
      frame = 0
      const batch = buffer
      buffer = []
      setEvents((prev) => [...prev, ...batch].sort((a, b) => a.seq - b.seq))
    }
    const source = new EventSource(url)
    const onEvent = (msg: MessageEvent) => {
      const e = JSON.parse(msg.data) as Ev
      if (seen.has(e.seq)) return
      seen.add(e.seq)
      buffer.push(e)
      if (!frame) frame = setTimeout(flush, 30)
    }
    EVENTS.forEach((name) => source.addEventListener(name, onEvent))
    source.onopen = () => setConnected(true)
    source.onerror = () => {
      setConnected(false)
      if (source.readyState === EventSource.CLOSED) setGone(true)
    }
    return () => {
      if (frame) clearTimeout(frame)
      source.close()
    }
  }, [url])

  return { events, connected, gone }
}

/** Poll a loader every few seconds; the rail's lists, the pool, the workspace. */
export function usePolled<T>(load: () => Promise<T>, deps: unknown[], every = 5000): [T | null, () => void] {
  const [value, setValue] = useState<T | null>(null)
  const [tick, setTick] = useState(0)
  useEffect(() => {
    let alive = true
    const run = () => load().then((v) => alive && setValue(v)).catch(() => {})
    run()
    const timer = setInterval(run, every)
    return () => { alive = false; clearInterval(timer) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick])
  return [value, () => setTick((t) => t + 1)]
}

/** "Edit & re-run": the run to replace and its settings, carried to the start page. */
export type RerunDraft = { direction: string; params: Params; replaces: string }
const RERUN_KEY = 'autofe.rerun'
export const rerunDraft = {
  set(draft: RerunDraft) {
    try { sessionStorage.setItem(RERUN_KEY, JSON.stringify(draft)) } catch { /* private mode */ }
  },
  take(): RerunDraft | null {
    try {
      const raw = sessionStorage.getItem(RERUN_KEY)
      sessionStorage.removeItem(RERUN_KEY)
      return raw ? JSON.parse(raw) : null
    } catch {
      return null
    }
  },
}
