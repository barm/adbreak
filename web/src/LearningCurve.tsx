import { useState, type CSSProperties } from "react";
import { Stack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";

export type CurvePoint = { k: number; runs: number; hours: number; log_loss: number[]; f1: number[]; auc: number[] };
export type Curve = { computed_at: string; recordings: number; sets: number; fits: number; points: CurvePoint[] };

const W = 420, H = 230, M = { l: 44, r: 46, t: 8, b: 34 };
const series = "var(--color-data-categorical-blue, var(--color-accent))";
const styles: Record<string, CSSProperties> = {
  wrap: { position: "relative" },
  tip: {
    position: "absolute", pointerEvents: "none", background: "var(--color-background-card)", border: "1px solid var(--color-border)",
    borderRadius: "var(--radius-2)", padding: "var(--spacing-1) var(--spacing-2)", whiteSpace: "nowrap", zIndex: 1,
  },
};

/** One measure of the learning curve against the number of recordings trained on: median line, middle-half band. */
function Panel({ curve, metric, title, note, fmt, log, domain }: {
  curve: Curve; metric: "log_loss" | "f1"; title: string; note: string; fmt: (v: number) => string; log: boolean; domain: [number, number];
}) {
  const [hover, setHover] = useState<CurvePoint | null>(null);
  const pts = curve.points;
  const kmax = Math.max(...pts.map((p) => p.k));
  const x = (k: number) => M.l + ((k - 1) / Math.max(1, kmax - 1)) * (W - M.l - M.r);
  const f = log ? Math.log10 : (v: number) => v;
  const [d0, d1] = domain.map(f);
  const y = (v: number) => M.t + (1 - (f(Math.min(Math.max(v, domain[0]), domain[1])) - d0) / (d1 - d0)) * (H - M.t - M.b);
  const ticks = log ? [0.05, 0.1, 0.2, 0.5, 1, 2] : [0.5, 0.6, 0.7, 0.8, 0.9, 1];
  const band = [...pts.map((p) => `${x(p.k)},${y(p[metric][2])}`), ...[...pts].reverse().map((p) => `${x(p.k)},${y(p[metric][0])}`)].join(" ");
  const last = pts[pts.length - 1];
  return (
    <Stack gap={1}>
      <Text type="label">{title}</Text>
      <Text type="supporting" color="secondary">{note}</Text>
      <Stack style={styles.wrap}>
        <svg width={W} height={H} role="img" aria-label={`${title} by recordings trained on`}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="var(--color-border)" strokeWidth={1} />
              <text x={M.l - 6} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--color-text-secondary)">{fmt(t)}</text>
            </g>
          ))}
          {pts.map((p) => (
            <text key={p.k} x={x(p.k)} y={H - M.b + 16} textAnchor="middle" fontSize={11} fill="var(--color-text-secondary)">{p.k}</text>
          ))}
          <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle" fontSize={11} fill="var(--color-text-secondary)">recordings trained on</text>
          <polygon points={band} fill={series} opacity={0.18} />
          <polyline points={pts.map((p) => `${x(p.k)},${y(p[metric][1])}`).join(" ")} fill="none" stroke={series} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          {pts.map((p) => <circle key={p.k} cx={x(p.k)} cy={y(p[metric][1])} r={4} fill={series} stroke="var(--color-background-surface)" strokeWidth={2} />)}
          <text x={x(last.k) + 8} y={y(last[metric][1]) + 4} fontSize={12} fill="var(--color-text-primary)">{fmt(last[metric][1])}</text>
          {hover && <line x1={x(hover.k)} x2={x(hover.k)} y1={M.t} y2={H - M.b} stroke="var(--color-border-emphasized)" strokeWidth={1} />}
          <rect
            x={M.l - 10} y={M.t} width={W - M.l - M.r + 20} height={H - M.t - M.b} fill="transparent"
            onPointerMove={(e) => {
              const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
              const k = Math.round(((e.clientX - r.left - M.l) / (W - M.l - M.r)) * (kmax - 1) + 1);
              setHover(pts.find((p) => p.k === Math.max(1, Math.min(kmax, k))) ?? null);
            }}
            onPointerLeave={() => setHover(null)}
          />
        </svg>
        {hover && (
          <Stack style={{ ...styles.tip, left: Math.min(x(hover.k) + 12, W - 190), top: y(hover[metric][1]) + 8 }} gap={0.5}>
            <Text type="label">{`${hover.k} recording${hover.k > 1 ? "s" : ""} (~${hover.hours.toFixed(1)} h labelled)`}</Text>
            <Text type="supporting">{`median ${fmt(hover[metric][1])} · middle half ${fmt(hover[metric][0])}–${fmt(hover[metric][2])}`}</Text>
            <Text type="supporting" color="secondary">{`${hover.runs} runs`}</Text>
          </Stack>
        )}
      </Stack>
    </Stack>
  );
}

/** The commercial-break model trained on k recordings, scored on the ones it never saw. */
export function LearningCurve({ curve }: { curve: Curve }) {
  return (
    <Stack direction="horizontal" gap={6} wrap="wrap">
      <Panel curve={curve} metric="log_loss" title="Held-out log loss" note="Lower is better · log scale" fmt={(v) => (v < 0.1 ? v.toFixed(3) : v.toFixed(2))} log domain={[0.04, 3]} />
      <Panel curve={curve} metric="f1" title="Held-out F1, per second" note="Higher is better · at the 0.5 threshold" fmt={(v) => `${Math.round(v * 100)}%`} log={false} domain={[0.5, 1]} />
    </Stack>
  );
}
