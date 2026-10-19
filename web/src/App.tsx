import { useEffect, useState } from 'react'
import { api, usePolled } from './api'
import { Rail } from './components/Rail'
import { SetupPage } from './components/SetupPage'
import { NewDirection } from './components/NewDirection'
import { RunPage } from './components/RunPage'
import { EvaluatePage } from './components/EvaluatePage'
import { ResultsPage } from './components/ResultsPage'
import { RequestsPage } from './components/RequestsPage'
import { InferencePage } from './components/InferencePage'
import { LoggingBoard } from './components/LoggingBoard'
import { Intro } from './components/Intro'
import s from './App.module.css'

// The journey: 1. Setup - point at the data, confirm each source's linkage;
// 2. Discover - run directions that generate verified features;
// 3. Evaluate - the feature set: pick single features and combinations, judge
//    them on test; the request set gathers every kept data request; Results lists
//    every variant ever evaluated, with its details; Inference scores the target
//    cases with base and with base + new features.
export type Page = 'setup' | 'discover' | 'evaluate' | 'requests' | 'results' | 'inference'
export type Route = { page: Page; id: string | null; direction: string | null }

// The route lives in the hash - #/discover/<run>, #/results/<evaluation>[.<variant>] - so a
// reload or a shared link reopens the same view. #/discover?direction=<text> opens a new
// direction with its text filled in - a link from another page can start one.
function readRoute(): Route {
  const hash = window.location.hash
  const queryIndex = hash.indexOf('?')
  const path = queryIndex < 0 ? hash : hash.slice(0, queryIndex)
  const query = new URLSearchParams(queryIndex < 0 ? '' : hash.slice(queryIndex + 1))
  const [, page, id] = path.split('/')
  const known: Page[] = ['setup', 'discover', 'evaluate', 'requests', 'results', 'inference']
  return {
    page: known.includes(page as Page) ? (page as Page) : 'setup',
    id: id || null,
    direction: page === 'discover' && !id ? query.get('direction') : null,
  }
}

// The opening plays once per browser session; the logo plays it again.
const INTRO_KEY = 'autofe.intro.seen'
const introSeen = () => { try { return sessionStorage.getItem(INTRO_KEY) === '1' } catch { return false } }

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
  const [board, setBoard] = useState(false)
  const [intro, setIntro] = useState(() => !introSeen())
  // Entering from the opening, the platform comes up into focus beneath its dive.
  const [emerging, setEmerging] = useState(false)
  const closeIntro = () => {
    try { sessionStorage.setItem(INTRO_KEY, '1') } catch { /* private mode */ }
    setIntro(false)
    setTimeout(() => setEmerging(false), 400)
  }

  // An evaluation used to open under Evaluate; it now opens beside the results.
  useEffect(() => {
    if (route.page === 'evaluate' && route.id) navigate('results', route.id)
  }, [route])

  // Opening the app with nothing selected while a direction runs shows that run.
  useEffect(() => {
    if (!window.location.hash && workspace?.active_run) navigate('discover', workspace.active_run)
  }, [workspace?.active_run])

  return (
    <div className={`${s.shell} ${emerging ? s.emerging : ''}`}>
      <Rail route={route} workspace={workspace} runs={runs ?? []} evaluations={evaluations ?? []}
            onBoard={() => setBoard(true)} onLogo={() => setIntro(true)} />
      <main className={s.main}>
        {route.page === 'setup' && (
          <SetupPage workspace={workspace} onChange={refreshWorkspace} />
        )}
        {route.page === 'discover' && !route.id && (
          <NewDirection key={route.direction} workspace={workspace}
                        initialDirection={route.direction}
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
        {route.page === 'requests' && <RequestsPage />}
        {route.page === 'results' && <ResultsPage selected={route.id} />}
        {route.page === 'inference' && <InferencePage />}
      </main>
      {board && <LoggingBoard onClose={() => setBoard(false)} />}
      {intro && <Intro onEnter={closeIntro} onLeave={() => setEmerging(true)} />}
    </div>
  )
}
