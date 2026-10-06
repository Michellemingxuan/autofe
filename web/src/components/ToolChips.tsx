import type { ToolUse } from '../types'
import s from './ToolChips.module.css'

/**
 * The tools a step called, in order. Repeats of one tool fold into "×n"; a call
 * that was sent back (refused, nothing spent) is marked. Calls one model response
 * released together say so on hover.
 */
export function ToolChips({ tools, tone = 'light' }: { tools: ToolUse[]; tone?: 'light' | 'dark' }) {
  if (!tools.length) return null
  const groups: { tool: string; ok: boolean; parallel: number; n: number }[] = []
  for (const t of tools) {
    const last = groups[groups.length - 1]
    if (last && last.tool === t.tool && last.ok === t.ok && last.parallel === t.parallel) last.n += 1
    else groups.push({ tool: t.tool, ok: t.ok, parallel: t.parallel, n: 1 })
  }
  return (
    <div className={`${s.row} ${s[tone]}`}>
      {groups.map((g, i) => (
        <span key={i} className={`${s.chip} ${g.ok ? '' : s.refused}`}
              title={`${g.tool}${g.n > 1 ? ` ×${g.n}` : ''}${g.parallel > 1 ? ` - one of ${g.parallel} calls released in parallel` : ''}${g.ok ? '' : ' - sent back to fix, nothing spent'}`}>
          {!g.ok && <span className={s.mark} aria-label="sent back">↩</span>}
          {g.tool}{g.n > 1 && <span className={s.count}>×{g.n}</span>}
        </span>
      ))}
    </div>
  )
}
