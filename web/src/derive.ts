import type {
  ApprovalCard, CodeCard, DataRequest, EvalView, Ev, LedgerRow, RefusedTry, RunView, Step, StepKind,
  ToolUse, TraceItem,
} from './types'

// Runs from before the rename call it request_data_pull.
const isRequest = (tool: string) => tool === 'screen_request' || tool === 'request_data_pull'

const parse = (text: string): any => {
  try { return JSON.parse(text) } catch { return {} }
}

/**
 * Fold a run's events into steps - the stages the trace groups and the
 * timeline lists. A step opens when the agent starts something with its own
 * outcome (a linkage, a data request, a feature) and closes when that tool
 * returns. Talk and exploration between them gather into an open step that
 * becomes the next intent once the agent screens a feature from it - so an
 * intent shows the reasoning and probes that led to it, not just its result.
 * Pure, so a replay always derives the same view.
 */
export function deriveRun(events: Ev[]): RunView {
  if (isL3(events)) return deriveL3(events)
  const view: RunView = {
    direction: '', K: 0, params: null, startTs: 0, steps: [], ledger: [], requests: [], skills: [],
    pending: [], status: 'idle', stoppedBecause: null, codeCount: 0,
  }
  const codes = new Map<string, CodeCard>()
  const approvals = new Map<string, ApprovalCard>()
  // The open step; a holder, since the helpers below reassign it.
  const at: { step: Step | null } = { step: null }
  const calls = new Map<string, ToolUse>()
  const sentBack = new Map<string, Step>()          // source name -> its request step, sent back
  let batch = 0
  let lastWasCall = false
  let lastTs = 0

  const close = (status?: Step['status']) => {
    if (!at.step) return
    if (status) at.step.status = status
    else if (at.step.status === 'running' || at.step.status === 'waiting') at.step.status = 'done'
    at.step.endTs = lastTs
    at.step = null
  }
  const open = (kind: StepKind, title: string): Step => {
    close()
    at.step = { id: view.steps.length, kind, title, status: 'running', items: [], tools: [],
                startTs: lastTs, endTs: lastTs }
    view.steps.push(at.step)
    return at.step
  }
  const here = (): Step => at.step ?? open('explore', '')

  for (const e of events) {
    // Runs from before evaluation moved to its own page carry its events;
    // they belong to the Evaluate page, not to the run's steps.
    if (e.event.startsWith('eval_') || e.intent === 'evaluate') continue
    lastTs = e.ts
    if (at.step) at.step.endTs = e.ts
    if (e.event !== 'tool_started') lastWasCall = false
    switch (e.event) {
      case 'run_started':
        view.direction = e.direction
        view.K = e.K
        view.params = e.params ?? null
        view.quota = e.quota ?? undefined
        view.startTs = e.ts
        view.status = 'running'
        break

      case 'agent_message': {
        // Words after the run ended are its closing words: they belong to the summary.
        const summary = view.status === 'done' && !at.step
          ? [...view.steps].reverse().find((st) => st.kind === 'summary') : undefined
        ;(summary ?? here()).items.push({ kind: 'message', seq: e.seq, text: e.text })
        break
      }

      case 'tool_started': {
        const args = parse(e.args)
        if (e.tool === 'propose_linkage') open('linkage', `Linkage · ${args.source ?? ''}`)
        else if (isRequest(e.tool)) {
          // A recorded request arrives before its call is flushed: the step
          // that holds it is this call's step, not a new one.
          const name = args.source_name ?? ''
          const retry = sentBack.get(name)                 // a retry of a try the screen sent back
          if (retry) {
            close()
            at.step = retry
            retry.status = 'running'
            sentBack.delete(name)
          } else if (!(at.step?.request && at.step.request.source_name === name)) {
            open('l3', name)
          }
        } else if (e.tool === 'challenge_request') {
          // The verdict is recorded before this call is flushed - same step.
          if (at.step?.challengeOf !== args.intent) {
            const step = open('challenge', `Challenge · ${args.intent ?? ''}`)
            step.challengeOf = args.intent
          }
        } else if (e.tool === 'report_findings' || e.tool === 'finish') {
          // Likewise the run may already be done - its summary step exists.
          if (!view.steps.some((st) => st.kind === 'summary')) open('summary', 'Summary')
        }
        else if (e.tool === 'screen_feature') {
          // The call is often announced after its result: a step already scored keeps it.
          const step = here()
          if (!step.row) {
            step.kind = 'intent'
            step.title = args.name ?? 'feature'
            step.level = args.level
            step.status = 'running'
          }
        } else if (e.tool !== 'run_probe' && e.tool !== 'load_skill' && e.tool !== 'brainstorm') {
          here().items.push({ kind: 'tool', seq: e.seq, tool: e.tool, args: e.args })
        }
        // The call on the step it belongs to; calls of one model response share a batch.
        if (!lastWasCall) batch += 1
        lastWasCall = true
        const use: ToolUse = { call_id: e.call_id, tool: e.tool, batch, parallel: 1, ok: true }
        calls.set(e.call_id, use)
        // The report ends the run before its call is announced: it belongs to the summary.
        const summary = e.tool === 'report_findings'
          ? [...view.steps].reverse().find((st) => st.kind === 'summary') : undefined
        if (summary) summary.tools!.push(use)
        else if (e.tool !== 'brainstorm') here().tools!.push(use)   // brainstorm: placed once its outcome is known
        break
      }

      case 'tool_completed': {
        const out = parse(e.output)
        const use = calls.get(e.call_id)
        if (use) use.ok = !(out.ok === false || (out.error && !out.recorded && !out.intent))
        if (use?.tool === 'brainstorm') {
          // Accepted, it belongs to the ideas it recorded; sent back, to the exploration.
          const ideas = [...view.steps].reverse().find((st) => st.kind === 'ideas' &&
            !st.tools!.some((u) => u.tool === 'brainstorm'))
          ;(use.ok && ideas ? ideas : here()).tools!.push(use)
        }
        if (!at.step) break
        const step: Step = at.step
        if (e.tool === 'propose_linkage' || isRequest(e.tool)) {
          if (out.error && !out.approved) step.note = String(out.error).slice(0, 220)
          const passed = out.ok || out.approved || out.recorded
          // A request the screen sent back waits for its retry, in the same step.
          if (!passed && isRequest(e.tool)) sentBack.set(step.title, step)
          close(passed ? 'done' : isRequest(e.tool) ? 'sent_back' : 'failed')
        } else if (e.tool === 'screen_feature') {
          if (out.error && !out.intent) {
            // Refused before it ran (budget, naming, missing skill): no intent spent.
            step.note = String(out.error).slice(0, 160)
            step.kind = 'explore'
            step.title = ''
            step.status = 'running'
          } else close()
        }
        break
      }

      case 'ideas_recorded': {
        const step = open('ideas', `${e.ideas.length} ideas · ${new Set(e.ideas.map((i: any) => i.lens)).size} lenses`)
        step.items.push({ kind: 'ideas', seq: e.seq, focus: e.focus ?? [], ideas: e.ideas })
        close('done')
        break
      }

      case 'ideas_sent_back':
        here().items.push({ kind: 'note', seq: e.seq, tone: 'info',
          text: `Ideas sent back to fix (try ${e.attempt}): ${e.error}` })
        break

      case 'skill_loaded':
        if (!view.skills.includes(e.name)) view.skills.push(e.name)
        here().items.push({ kind: 'skill', seq: e.seq, name: e.name })
        break

      case 'code_status': {
        const prev = codes.get(e.code_id)
        if (!prev) {
          const card: CodeCard = {
            kind: 'code', seq: e.seq, code_id: e.code_id, intent: e.intent, mode: e.mode,
            level: e.level, title: e.title, code: e.code ?? '', state: e.state,
          }
          codes.set(e.code_id, card)
          here().items.push(card)
          view.codeCount += 1
        } else {
          Object.assign(prev, { state: e.state, stdout: e.stdout, error: e.error,
                                result: e.result, elapsed_s: e.elapsed_s })
        }
        break
      }

      case 'approval_required': {
        const card: ApprovalCard = { kind: 'approval', seq: e.seq, req_id: e.req_id,
                                     type: e.kind, payload: e }
        approvals.set(e.req_id, card)
        const step = here()
        step.items.push(card)
        step.status = 'waiting'
        break
      }

      case 'approval_resolved': {
        const card = approvals.get(e.req_id)
        if (card) card.resolved = { approved: e.approved, note: e.note }
        if (at.step && at.step.status === 'waiting') at.step.status = 'running'
        break
      }

      case 'feature_screened': {
        const row: LedgerRow = {
          intent: e.intent, name: e.name, description: e.description, level: e.level,
          code_id: e.code_id, sources: e.sources ?? [], delta: e.delta, capture_gain: e.capture_gain ?? null,
          base_score: e.base_score, coverage: e.coverage, verified: e.verified, reason: e.reason,
        }
        view.ledger.push(row)
        const step = here()
        step.kind = 'intent'
        step.title = e.name
        step.level = e.level
        step.row = row
        step.status = e.verified ? 'verified' : 'rejected'
        break
      }

      case 'intent_deleted': {
        const row = view.ledger.find((r) => r.name === e.name)
        if (row) row.deleted = true
        break
      }

      case 'data_request': {
        const request: DataRequest = {
          intent: e.intent, source_name: e.source_name, gap: e.gap, features: e.features ?? '',
          sql: e.sql, tables: e.tables ?? [], columns: e.columns ?? [], status: e.status,
        }
        view.requests.push(request)
        const step = at.step && !at.step.request && at.step.kind !== 'summary' && at.step.kind !== 'stage'
          ? at.step : open('l3', `Data request · ${e.source_name}`)
        step.kind = 'l3'
        step.title = e.source_name
        step.request = request
        break
      }

      case 'request_challenged': {
        const request = view.requests.find((r) => r.intent === e.intent)
        if (request) {
          request.status = e.status
          request.refunded = !!e.refunded
          request.challenge = { verdict: e.verdict, reasoning: e.reasoning, columns: e.columns ?? [],
                                code: e.code ?? '', code_ok: e.code_ok, code_error: e.code_error,
                                note: e.note }
        }
        // Its own step: reuse the open one only if it is this verdict's call, or
        // the challenger's look-around before it - never another verdict's step.
        const reusable = at.step && (at.step.challengeOf === e.intent ||
          (!at.step.challengeOf && at.step.kind === 'explore'))
        const step: Step = reusable && at.step ? at.step : open('challenge', `Challenge · ${e.intent}`)
        step.kind = 'challenge'
        step.title = `${e.intent} · ${e.source_name}`
        step.challengeOf = e.intent
        step.request = request
        step.status = e.status === 'dropped' ? 'rejected' : 'verified'
        break
      }

      case 'stage_done': {
        // The proposer's own account. It stays open for the proposer's closing
        // words, which arrive just after; the challenge stage closes it.
        const step = open('summary', 'Proposals')
        step.summary = e.summary || ''
        const l3 = view.params?.levels?.length === 1 && view.params.levels[0] === 'L3'
        if (l3 && !view.requests.length) {
          step.items.push({ kind: 'note', seq: e.seq, tone: 'error',
            text: 'No data requests were proposed. The agent explains why below - if it did not '
                + 'consider the unused CAS variables, try a more specific direction.' })
        }
        break
      }

      case 'stage_started': {
        close('done')
        const step = open('stage', 'Can each be built from current data?')
        step.note = 'A challenger reviews every proposal'
        close('done')
        break
      }

      case 'agent_nudged':
        here().items.push({ kind: 'note', seq: e.seq, tone: 'info',
          text: `The agent stopped without reporting - asked to carry on (${e.attempt} of 3).` })
        break

      case 'source_detected':
        here().items.push({ kind: 'note', seq: e.seq, tone: 'info',
          text: `New source detected: ${e.name}${e.usable ? '' : ' (schema only)'}` })
        break

      case 'run_done': {
        const step = at.step && at.step.kind === 'summary' ? at.step : open('summary', 'Summary')
        step.summary = e.summary || ''
        const l3 = view.params?.levels?.length === 1 && view.params.levels[0] === 'L3'
        if (l3 && !view.requests.length) {
          step.items.push({ kind: 'note', seq: e.seq, tone: 'error',
            text: 'No data requests were proposed. The agent explains why below - if it did not '
                + 'consider the unused CAS variables, try a more specific direction.' })
        }
        step.note = `${e.stopped_because} · ${e.verified.length} verified of ${e.intents_used}/${e.K}`
        close('done')
        view.status = 'done'
        view.stoppedBecause = e.stopped_because
        break
      }

      case 'run_error':
        here().items.push({ kind: 'note', seq: e.seq, tone: 'error', text: e.error })
        close('failed')
        view.status = 'error'
        break
    }
  }

  view.pending = [...approvals.values()].filter((a) => !a.resolved)
  if (view.status === 'running' && view.pending.length) view.status = 'waiting'
  markUsedIdeas(view)
  nameLookSteps(view)
  // How many calls each model response released together.
  const sizes = new Map<number, number>()
  for (const use of calls.values()) sizes.set(use.batch, (sizes.get(use.batch) ?? 0) + 1)
  for (const use of calls.values()) use.parallel = sizes.get(use.batch) ?? 1
  // Code cards update in place; hand React fresh step objects so the
  // memoised panes notice.
  view.steps = view.steps.map((s) => ({ ...s, items: [...s.items], tools: (s.tools ?? []).map((u) => ({ ...u })) }))
  return view
}

export function deriveEval(events: Ev[]): EvalView {
  const view: EvalView = {
    status: 'idle', startTs: 0, lastTs: 0, features: [], combinations: {}, codes: [],
    stages: [], logs: [], rows: [], verdicts: [], capturePercents: [], shapRanks: {}, featureStats: {},
  }
  const codes = new Map<string, CodeCard>()
  for (const e of events) {
    view.lastTs = e.ts
    switch (e.event) {
      case 'eval_started':
        view.status = 'running'
        view.startTs = e.ts
        view.features = e.features
        view.combinations = e.combinations
        break
      case 'code_status': {
        const prev = codes.get(e.code_id)
        if (!prev) {
          const card: CodeCard = { kind: 'code', seq: e.seq, code_id: e.code_id, intent: e.intent,
                                   mode: e.mode, level: e.level, title: e.title, code: e.code ?? '',
                                   state: e.state }
          codes.set(e.code_id, card)
          view.codes.push(card)
        } else {
          Object.assign(prev, { state: e.state, error: e.error, elapsed_s: e.elapsed_s })
        }
        break
      }
      case 'eval_status':
        view.stages = e.stages
        break
      case 'eval_log':
        view.logs.push({ seq: e.seq, ts: e.ts, message: e.message, level: e.level ?? 'info' })
        break
      case 'eval_done':
        view.status = 'done'
        view.rows = e.comparison
        view.verdicts = e.verdicts
        view.outputDir = e.output_dir
        view.capturePercents = e.capture_percents ?? []
        view.shapRanks = e.shap_ranks ?? {}
        view.featureStats = e.feature_stats ?? {}
        break
      case 'eval_error':
        view.status = 'error'
        view.error = e.error
        break
      case 'variant_removed':                 // taken off the results by the user
        view.rows = view.rows.filter((r) => r.variant !== e.variant)
        break
    }
  }
  view.codes = view.codes.map((c) => ({ ...c }))
  return view
}

const isL3 = (events: Ev[]) => {
  const started = events.find((e) => e.event === 'run_started')
  const levels = started?.params?.levels
  return Array.isArray(levels) && levels.length === 1 && levels[0] === 'L3'
}

/**
 * A data-request run, one step per request. Each request - R1, R2, ... - holds
 * everything that happened to it: the words that led to it, a try validation
 * sent back, the recorded proposal and its SQL, the agent's challenge of it, and
 * whether it was kept or dropped. Exploration between requests is its own
 * step. Steps are found by the request's intent and by each tool call's id, not
 * by the order events arrive in - several calls of one model response run before
 * any of them is announced, so arrival order does not say which call is whose.
 */
export function deriveL3(events: Ev[]): RunView {
  const view: RunView = {
    direction: '', K: 0, params: null, startTs: 0, steps: [], ledger: [], requests: [], skills: [],
    pending: [], status: 'idle', stoppedBecause: null, codeCount: 0,
  }
  const byIntent = new Map<string, Step>()
  const calls = new Map<string, { use: ToolUse; step: Step | null; args: any }>()
  const codes = new Map<string, CodeCard>()
  const refusedBySource = new Map<string, { tries: RefusedTry[]; uses: ToolUse[] }>()
  let explore: Step | null = null
  let summary: Step | null = null
  let ideasStep: Step | null = null
  let words: TraceItem[] = []           // narration waiting for the step it leads to
  let batch = 0
  let lastWasCall = false
  let lastTs = 0

  const push = (kind: StepKind, title: string): Step => {
    const step: Step = { id: view.steps.length, kind, title, status: 'running', items: [], tools: [],
                         startTs: lastTs, endTs: lastTs }
    view.steps.push(step)
    return step
  }
  const withWords = (step: Step) => {            // the narration, in the order it was said
    step.items.push(...words.filter((w) => !step.items.includes(w)))
    words = []
    return step
  }
  const exploreStep = () => withWords(explore ?? (explore = push('explore', '')))
  const summaryStep = () => withWords(summary ?? (summary = push('summary', 'Summary')))

  for (const e of events) {
    if (e.event.startsWith('eval_')) continue
    lastTs = e.ts
    if (e.event !== 'tool_started') lastWasCall = false
    switch (e.event) {
      case 'run_started':
        view.direction = e.direction
        view.K = e.K
        view.params = e.params ?? null
        view.quota = e.quota ?? undefined
        view.startTs = e.ts
        view.status = 'running'
        break

      case 'agent_message':
        words.push({ kind: 'message', seq: e.seq, text: e.text })
        break

      case 'data_request': {
        const request: DataRequest = {
          intent: e.intent, source_name: e.source_name, gap: e.gap, features: e.features ?? '',
          sql: e.sql, tables: e.tables ?? [], columns: e.columns ?? [], status: e.status,
        }
        view.requests.push(request)
        const step = withWords(push('l3', e.source_name))
        step.request = request
        const earlier = refusedBySource.get(e.source_name)
        if (earlier) {                       // tries of this pull that validation sent back
          step.refused = earlier.tries
          step.tools!.push(...earlier.uses)
          refusedBySource.delete(e.source_name)
        }
        byIntent.set(e.intent, step)
        explore = null
        break
      }

      case 'ideas_recorded': {
        const step = withWords(push('ideas', `${e.ideas.length} ideas · ${new Set(e.ideas.map((i: any) => i.lens)).size} lenses`))
        step.items.push({ kind: 'ideas', seq: e.seq, focus: e.focus ?? [], ideas: e.ideas })
        step.status = 'done'
        ideasStep = step
        explore = null
        break
      }

      case 'request_challenged': {
        const step = byIntent.get(e.intent)
        const request = step?.request
        if (request) {
          request.status = e.status
          request.refunded = !!e.refunded
          request.challenge = { verdict: e.verdict, reasoning: e.reasoning, columns: e.columns ?? [],
                                code: e.code ?? '', code_ok: e.code_ok, code_error: e.code_error,
                                note: e.note }
        }
        if (step) step.status = e.status === 'dropped' ? 'rejected' : 'verified'
        break
      }

      case 'tool_started': {
        if (!lastWasCall) batch += 1
        lastWasCall = true
        const args = parse(e.args)
        const use: ToolUse = { call_id: e.call_id, tool: e.tool, batch, parallel: 1, ok: true }
        let step: Step | null = null
        if (isRequest(e.tool)) {
          // The recorded proposal with this exact SQL - else a try that was refused.
          step = [...byIntent.values()].find((st) => st.request?.source_name === args.source_name &&
            (st.request?.sql ?? '').trim() === String(args.sql ?? '').trim() &&
            !st.tools!.some((t) => t.tool === e.tool && t.ok)) ?? null
          if (step) withWords(step)
        } else if (e.tool === 'challenge_request') {
          step = byIntent.get(args.intent) ?? null
          if (step) { withWords(step); explore = null }
        } else if (e.tool === 'brainstorm' && ideasStep && !ideasStep.tools!.some((t) => t.ok)) {
          step = ideasStep
        } else if (e.tool === 'report_findings') {
          step = summaryStep()
        } else {
          step = exploreStep()
        }
        if (step) step.tools!.push(use)
        calls.set(e.call_id, { use, step, args })
        break
      }

      case 'tool_completed': {
        const call = calls.get(e.call_id)
        if (!call) break
        const out = parse(e.output)
        call.use.ok = !(out.ok === false || (out.error && !out.recorded))
        if (!call.step && isRequest(call.use.tool)) {
          const name = call.args.source_name ?? '?'
          const entry = refusedBySource.get(name) ?? { tries: [], uses: [] }
          entry.tries.push({ sql: call.args.sql ?? '', error: String(out.error ?? e.output) })
          entry.uses.push(call.use)
          refusedBySource.set(name, entry)
        }
        if (call.use.tool === 'report_findings' && out.ok === false && call.step) {
          call.step.items.push({ kind: 'note', seq: e.seq, tone: 'info', text: String(out.error) })
        }
        break
      }

      case 'code_status': {
        const prev = codes.get(e.code_id)
        if (prev) {
          Object.assign(prev, { state: e.state, stdout: e.stdout, error: e.error,
                                result: e.result, elapsed_s: e.elapsed_s })
          break
        }
        const card: CodeCard = { kind: 'code', seq: e.seq, code_id: e.code_id, intent: e.intent,
                                 mode: e.mode, level: e.level, title: e.title, code: e.code ?? '',
                                 state: e.state }
        codes.set(e.code_id, card)
        view.codeCount += 1
        // A challenge's construction is shown with its verdict, and a reused linkage is
        // plumbing for one; only a probe is exploration.
        if (e.mode === 'probe' && !byIntent.has(e.intent)) exploreStep().items.push(card)
        break
      }

      case 'ideas_sent_back':
        exploreStep().items.push({ kind: 'note', seq: e.seq, tone: 'info',
          text: `Ideas sent back to fix (try ${e.attempt}): ${e.error}` })
        break

      case 'agent_nudged':
        exploreStep().items.push({ kind: 'note', seq: e.seq, tone: 'info',
          text: `The agent stopped without reporting - asked to carry on (${e.attempt} of 3).` })
        break

      case 'source_detected':
        exploreStep().items.push({ kind: 'note', seq: e.seq, tone: 'info',
          text: `New source detected: ${e.name}${e.usable ? '' : ' (schema only)'}` })
        break

      case 'run_done': {
        const step = summaryStep()
        step.summary = e.summary || ''
        step.note = e.stopped_because
        step.status = 'done'
        view.status = 'done'
        view.stoppedBecause = e.stopped_because
        break
      }

      case 'run_error':
        exploreStep().items.push({ kind: 'note', seq: e.seq, tone: 'error', text: e.error })
        view.status = 'error'
        break
    }
  }

  // Tries refused and never fixed are steps of their own.
  for (const [name, entry] of refusedBySource) {
    const step = push('l3', name)
    step.refused = entry.tries
    step.tools!.push(...entry.uses)
    step.status = 'sent_back'
  }
  if (words.length) exploreStep()
  // The summary ends the run, whatever came after it (runs from the two-stage design
  // reported once per stage).
  if (summary) {
    view.steps = [...view.steps.filter((st) => st !== summary), summary]
    view.steps.forEach((st, i) => { st.id = i })
  }
  // How many calls each model response released together.
  const sizes = new Map<number, number>()
  for (const { use } of calls.values()) sizes.set(use.batch, (sizes.get(use.batch) ?? 0) + 1)
  for (const { use } of calls.values()) use.parallel = sizes.get(use.batch) ?? 1
  for (const step of view.steps) {
    if (step.kind === 'explore' || step.kind === 'summary' || step.kind === 'ideas') {
      step.status = step === view.steps[view.steps.length - 1] && view.status === 'running'
        ? 'running' : 'done'
    }
    step.startTs = Math.min(step.startTs, step.endTs)
  }
  markUsedIdeas(view)
  nameLookSteps(view)
  view.steps = view.steps.map((s) => ({ ...s, items: [...s.items], tools: [...(s.tools ?? [])] }))
  return view
}

// The tools that only look; run_probe is counted by its scripts.
const LOOK_TOOLS = new Set(['catalog', 'scope', 'sample_rows', 'shots'])

/** A "look at the data" step is titled by what it looked at: the tools, and its scripts. */
function nameLookSteps(view: RunView) {
  for (const step of view.steps) {
    if (step.kind !== 'explore' || step.title) continue
    const counts = new Map<string, number>()
    const names = (step.tools?.length ? step.tools.map((u) => u.tool)
      : step.items.flatMap((i) => (i.kind === 'tool' ? [i.tool] : [])))
      .filter((t) => LOOK_TOOLS.has(t))
    for (const t of names) counts.set(t, (counts.get(t) ?? 0) + 1)
    const scripts = step.items.filter((i) => i.kind === 'code').length
    const parts = [...counts].map(([t, n]) => (n > 1 ? `${t} ×${n}` : t))
    if (scripts) parts.push(`${scripts} script${scripts > 1 ? 's' : ''}`)
    step.title = parts.join(' · ') || 'notes'
  }
}

/** Which ideas went on to a proposal: a feature or a request under the idea's name. */
function markUsedIdeas(view: RunView) {
  const used: Record<string, string> = {}
  for (const r of view.ledger) used[r.name] = r.intent
  for (const r of view.requests) used[r.source_name] = r.intent
  for (const step of view.steps) {
    for (const item of step.items) if (item.kind === 'ideas') item.used = used
  }
}
