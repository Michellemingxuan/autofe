import { useState } from 'react'
import s from './ConfirmButton.module.css'

/**
 * A destructive action in two clicks, confirmed in place - no browser
 * dialog. The second click runs it; anything else backs out.
 */
export function ConfirmButton({ label, confirm, onConfirm, tone = 'light', disabled, title }: {
  label: string
  confirm: string
  onConfirm: () => Promise<unknown> | void
  tone?: 'light' | 'dark'
  disabled?: boolean
  title?: string
}) {
  const [asking, setAsking] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (!asking) {
    return (
      <button className={`${s.btn} ${s[tone]}`} disabled={disabled} title={title}
              onClick={(e) => { e.stopPropagation(); setAsking(true); setError(null) }}>
        {label}
      </button>
    )
  }
  return (
    <span className={`${s.ask} ${s[tone]}`} onClick={(e) => e.stopPropagation()}>
      <span className={s.question}>{error ?? confirm}</span>
      <button className={s.yes} disabled={busy} onClick={async () => {
        setBusy(true)
        try {
          await onConfirm()
          setAsking(false)
        } catch (err: any) {
          setError(err.message)
        } finally {
          setBusy(false)
        }
      }}>{busy ? '…' : 'Delete'}</button>
      <button className={s.no} onClick={() => setAsking(false)}>Cancel</button>
    </span>
  )
}
