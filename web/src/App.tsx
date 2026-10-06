import { useEffect, useState } from 'react'
import { api, usePolled } from './api'
import { Rail } from './components/Rail'
import { SetupPage } from './components/SetupPage'
import { NewDirection } from './components/NewDirection'
import { RunPage } from './components/RunPage'
import { EvaluatePage } from './components/EvaluatePage'
import { ResultsPage } from './components/ResultsPage'
import s from './App.module.css'

// The journey: 1. Setup - point at the data, confirm each source's linkage;
// 2. Discover - run directions that generate verified features;
// 3. Evaluate - the feature set: pick single features and combinations, judge
//    them on test; Results lists every variant ever evaluated, with its details.
export type Page = 'setup' | 'discover' | 'evaluate' | 'results'
export type Route = { page: Page; id: string | null }

// The route lives in the hash - #/discover/<run>, #/results/<evaluation>[.<variant>] - so a
// reload or a shared link reopens the same view.
function readRoute(): Route {
  const [, page, id] = window.location.hash.split('/')
  const known: Page[] = ['setup', 'discover', 'evaluate', 'results']
  return { page: known.includes(page as Page) ? (page as Page) : 'setup', id: id || null }
}

export function navigate(page: Page, id: string | null = null) {
  window.location.hash = `/${page}${id ? `/${id}` : ''}`
}

export function App() {
  const [route, setRoute] = useState<Route>(readRoute)
  useEffect(() => {
    const onHash = () => setRoute(readRoute())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const [workspace, refreshWorkspace] = usePolled(api.workspace, [], 4000)
  const [runs, refreshRuns] = usePolled(api.runs, [])
  const [evaluations, refreshEvals] = usePolled(api.evaluations, [])

  // An evaluation used to open under Evaluate; it now opens beside the results.
  useEffect(() => {
    if (route.page === 'evaluate' && route.id) navigate('results', route.id)
  }, [route])

  // Opening the app with nothing selected while a direction runs shows that run.
  useEffect(() => {
    if (!window.location.hash && workspace?.active_run) navigate('discover', workspace.active_run)
  }, [workspace?.active_run])

  return (
    <div className={s.shell}>
      <Rail route={route} workspace={workspace} runs={runs ?? []} evaluations={evaluations ?? []} />
      <main className={s.main}>
        {route.page === 'setup' && (
          <SetupPage workspace={workspace} onChange={refreshWorkspace} />
        )}
        {route.page === 'discover' && !route.id && (
          <NewDirection workspace={workspace}
                        busy={!!(workspace?.active_run || workspace?.active_linkage)}
                        onStarted={(id) => { refreshRuns(); navigate('discover', id) }} />
        )}
        {route.page === 'discover' && route.id && (
          <RunPage key={route.id} runId={route.id} onChange={refreshRuns}
                   onDeleted={() => { refreshRuns(); navigate('discover') }}
                   onRerun={() => navigate('discover')} />
        )}
        {route.page === 'evaluate' && (
          <EvaluatePage onStarted={(id) => { refreshEvals(); navigate('results', id) }} />
        )}
        {route.page === 'results' && <ResultsPage selected={route.id} />}
      </main>
    </div>
  )
}
