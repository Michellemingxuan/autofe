import { useEffect, useMemo, useState } from 'react'
import s from './Intro.module.css'

/*
 * The opening: what the agent does, drawn as layers in depth. At the back the
 * raw data - contours and ticks; then the model's population as a scan, a field
 * of rows; in front the feature planes, each candidate a marker. Curves carry
 * the signal from the rows the agent singles out to the features it builds.
 *
 * The planes are rectangles in 3D, projected - so a curve can run from a point on
 * one plane to a point on another, through the space between them.
 */

const W = 430, H = 300                     // a plane, in its own units
const CX = 455, CY = 320                   // where the scene's centre lands on screen
const YAW = 0.86, PITCH = -0.12, CAMERA = 950

type P3 = [number, number, number]

function project([x, y, z]: P3): [number, number] {
  const cy = Math.cos(YAW), sy = Math.sin(YAW)
  const X = x * cy + z * sy
  let Z = -x * sy + z * cy
  const cp = Math.cos(PITCH), sp = Math.sin(PITCH)
  const Y = y * cp - Z * sp
  Z = y * sp + Z * cp
  const k = CAMERA / (CAMERA + Z)
  return [CX + X * k, CY + Y * k]
}

/** A point on the plane at depth z, from (u, v) in 0..1. */
const on = (z: number, u: number, v: number): P3 => [-W / 2 + u * W, -H / 2 + v * H, z]
const pts = (ps: P3[]) => ps.map((p) => project(p).map((n) => n.toFixed(1)).join(',')).join(' ')
const path = (ps: P3[], close = false) =>
  ps.map((p, i) => `${i ? 'L' : 'M'}${project(p).map((n) => n.toFixed(1)).join(' ')}`).join(' ') + (close ? ' Z' : '')

/** A rounded rectangle on a plane, sampled so it bends with the projection. */
function frame(z: number, r = 16): P3[] {
  const out: P3[] = []
  const corners: [number, number, number][] = [
    [W / 2 - r, -H / 2 + r, -Math.PI / 2], [W / 2 - r, H / 2 - r, 0],
    [-W / 2 + r, H / 2 - r, Math.PI / 2], [-W / 2 + r, -H / 2 + r, Math.PI]]
  for (const [cx, cy, a0] of corners) {
    for (let i = 0; i <= 6; i++) {
      const a = a0 + (i / 6) * (Math.PI / 2)
      out.push([cx + r * Math.cos(a), cy + r * Math.sin(a), z])
    }
  }
  return out
}

// A smooth field over the scan - the population's risk, as the model sees it.
const field = (u: number, v: number) =>
  0.5 + 0.22 * Math.sin(u * 7.1 + v * 2.3) + 0.18 * Math.cos(v * 8.3 - u * 3.1) + 0.1 * Math.sin((u + v) * 13)

// Seeded, so the picture is the same every time.
function rng(seed: number) {
  let x = seed
  return () => ((x = (x * 16807) % 2147483647) - 1) / 2147483646
}

const Z = { raw: -330, scan: -130, f1: 30, f2: 135, f3: 240 }
const LABELS = ['pay_to_spend_90d', 'Δ+0.043', 'L2', '#2', 'spend_vol', 'L1', '+6 cases', 'Δ+0.006', 'util_slope']

// Entering (~1.6s): the words step back, a light sweeps the layers, the camera
// dives through them into the data, and the workspace comes up out of the light.
const ENTER_MS = 1600

export function Intro({ onEnter, onLeave }: { onEnter: () => void; onLeave?: () => void }) {
  const [leaving, setLeaving] = useState(false)
  const enter = () => {
    if (leaving) return
    setLeaving(true)
    onLeave?.()                    // the platform starts to come up beneath the dive
    setTimeout(onEnter, ENTER_MS)
  }
  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape' || e.key === 'Enter') enter() }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  })

  const scene = useMemo(() => {
    const r = rng(7)
    // The raw data at the back: dashed rules, contour loops, scattered ticks.
    const rules = [0.2, 0.4, 0.6, 0.8].map((v) => path([on(Z.raw, 0.04, v), on(Z.raw, 0.96, v)]))
    const contours = [[0.3, 0.55, 0.12], [0.62, 0.38, 0.09], [0.7, 0.7, 0.07], [0.22, 0.28, 0.06], [0.48, 0.78, 0.05]]
      .flatMap(([u0, v0, rad]) => [1, 0.62].map((k) => {
        const ring: P3[] = []
        for (let i = 0; i <= 36; i++) {
          const a = (i / 36) * Math.PI * 2
          const wob = 1 + 0.18 * Math.sin(a * 3 + u0 * 9) + 0.1 * Math.cos(a * 5 + v0 * 7)
          ring.push(on(Z.raw, u0 + Math.cos(a) * rad * k * wob, v0 + Math.sin(a) * rad * k * wob * 1.3))
        }
        return path(ring, true)
      }))
    const ticks = Array.from({ length: 34 }, () => {
      const u = 0.06 + r() * 0.88, v = 0.06 + r() * 0.88
      return path([on(Z.raw, u, v), on(Z.raw, u + 0.012, v + 0.03)])
    })

    // The scan: a field of rows, brighter where the risk is.
    const dots: { p: [number, number]; level: number }[] = []
    for (let i = 0; i < 26; i++) for (let j = 0; j < 17; j++) {
      const u = 0.05 + (i / 25) * 0.9, v = 0.07 + (j / 16) * 0.86
      dots.push({ p: project(on(Z.scan, u, v)), level: field(u, v) })
    }
    // The rows the agent singles out, and the features they reach.
    const picks: [number, number][] = [[0.28, 0.32], [0.4, 0.62], [0.55, 0.25], [0.66, 0.55], [0.32, 0.8], [0.78, 0.4], [0.5, 0.45]]
    const ends: [number, number, number][] = [
      [Z.f3, 0.18, 0.2], [Z.f3, 0.3, 0.62], [Z.f2, 0.62, 0.3], [Z.f3, 0.78, 0.52],
      [Z.f2, 0.4, 0.82], [Z.f3, 0.66, 0.8], [Z.f2, 0.2, 0.45]]
    const curves = picks.map(([u, v], i) => {
      const [ze, ue, ve] = ends[i]
      const a = on(Z.scan, u, v), b = on(ze, ue, ve)
      const c1: P3 = [a[0] + 40 + r() * 60, a[1] + (r() - 0.5) * 120, Z.scan + 90]
      const c2: P3 = [b[0] - 40 - r() * 60, b[1] + (r() - 0.5) * 120, ze - 80]
      const line: P3[] = []
      for (let k = 0; k <= 40; k++) {
        const t = k / 40, m = 1 - t
        line.push([0, 1, 2].map((d) => m * m * m * a[d] + 3 * m * m * t * c1[d] + 3 * m * t * t * c2[d] + t * t * t * b[d]) as P3)
      }
      return { d: path(line), start: project(a), end: project(b) }
    })

    // The feature planes: candidate markers in rows, a few labelled.
    const markers = [Z.f1, Z.f2, Z.f3].flatMap((z, plane) =>
      Array.from({ length: 11 }, (_, i) => {
        const u = 0.1 + ((i * 0.37 + plane * 0.19) % 1) * 0.8
        const v = 0.12 + ((i * 0.53 + plane * 0.31) % 1) * 0.76
        const c = project(on(z, u, v))
        const size = 7 * (CAMERA / (CAMERA + z))
        return { c, size, plane, label: i % 4 === 1 ? LABELS[(i + plane * 3) % LABELS.length] : null }
      }))
    return { rules, contours, ticks, dots, curves, markers }
  }, [])

  return (
    <div className={`${s.intro} ${leaving ? s.leaving : ''}`} role="dialog" aria-label="AutoFE">
      <div className={s.backdrop} aria-hidden />
      <div className={s.glow} aria-hidden />
      <div className={s.sweep} aria-hidden />
      <div className={s.flash} aria-hidden />
      <svg className={s.scene} viewBox="0 0 900 640" aria-hidden>
        <defs>
          <linearGradient id="scanFill" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#123a7a" stopOpacity="0.95" />
            <stop offset="1" stopColor="#071c44" stopOpacity="0.95" />
          </linearGradient>
          <radialGradient id="dotGlow">
            <stop offset="0" stopColor="#ffffff" />
            <stop offset="1" stopColor="#7fb6ff" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* the raw data, at the back */}
        <g className={`${s.layer} ${s.l1}`}>
          <polygon points={pts(frame(Z.raw))} className={s.frameFaint} />
          {scene.rules.map((d, i) => <path key={i} d={d} className={s.dashed} />)}
          {scene.contours.map((d, i) => <path key={i} d={d} className={s.contour} />)}
          {scene.ticks.map((d, i) => <path key={i} d={d} className={s.tick} />)}
        </g>

        {/* the population, as a scan */}
        <g className={`${s.layer} ${s.l2}`}>
          <polygon points={pts(frame(Z.scan))} fill="url(#scanFill)" className={s.frameScan} />
          {scene.dots.map((d, i) => (
            <circle key={i} cx={d.p[0]} cy={d.p[1]} r={1 + d.level * 1.8}
                    fill={`rgba(${120 + d.level * 120}, ${170 + d.level * 70}, 255, ${0.25 + d.level * 0.6})`} />
          ))}
        </g>

        {/* the feature planes */}
        {[Z.f1, Z.f2, Z.f3].map((z, i) => (
          <g key={z} className={`${s.layer} ${s['l' + (3 + i)]}`}>
            <polygon points={pts(frame(z))} className={s.frame} />
          </g>
        ))}
        <g className={s.markers}>
          {scene.markers.map((m, i) => (
            <g key={i} style={{ animationDelay: `${0.8 + (i % 11) * 0.03 + m.plane * 0.1}s` }} className={s.marker}>
              <path d={`M${m.c[0]} ${m.c[1] - m.size} L${m.c[0] + m.size * 0.6} ${m.c[1]} L${m.c[0]} ${m.c[1] + m.size} L${m.c[0] - m.size * 0.6} ${m.c[1]} Z`} />
              {m.label && <text x={m.c[0] + m.size + 4} y={m.c[1] - m.size}>{m.label}</text>}
            </g>
          ))}
        </g>

        {/* the signal: from the rows singled out to the features built */}
        <g className={s.curves}>
          {scene.curves.map((c, i) => (
            <g key={i}>
              <path d={c.d} pathLength={1} className={s.curve} style={{ animationDelay: `${1.0 + i * 0.12}s` }} />
              <circle cx={c.start[0]} cy={c.start[1]} r={6} fill="url(#dotGlow)" className={s.pulse}
                      style={{ animationDelay: `${0.9 + i * 0.12}s` }} />
              <g className={s.hit} style={{ animationDelay: `${1.7 + i * 0.12}s` }}>
                <rect x={c.end[0] - 9} y={c.end[1] - 9} width={18} height={18} rx={4} />
                <path d={`M${c.end[0] - 3} ${c.end[1] - 3} L${c.end[0] + 4} ${c.end[1] + 4} L${c.end[0] - 3} ${c.end[1] + 4} Z`} />
              </g>
            </g>
          ))}
        </g>
      </svg>

      <div className={s.copy}>
        <div className={s.eyebrow}><span className={s.mark} />AUTOFE · MODEL AGENT</div>
        <h1 className={s.title}>Automated feature discovery and model enhancement.</h1>
        <p className={s.lead}>
          Starting with an analyst's intuition, the agent proposes ideas, builds and checks features, and proves them on data it never saw,
          down to the cases they would have caught sooner.
        </p>
        <button className={s.enter} onClick={enter} autoFocus>
          Enter the workspace <span aria-hidden>›</span>
        </button>
      </div>
    </div>
  )
}
