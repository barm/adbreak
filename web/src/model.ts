/** The model's data (`/api/model`) shared by the Recordings, Models and Runs pages, and `useModel` to load it, follow
 *  a run while one is going, and start one. */
import { useCallback, useEffect, useState } from "react";
import type { Curve } from "./LearningCurve";

export type Scored = { precision: number; recall: number; f1: number; breaks: number; found: number; matched: number; start_error: number; end_error: number; wrongly_skipped_per_hour: number; missed_per_hour: number };
export type Total = Scored & { source: string; kind: string };
export type RecScore = { id: string; title: string; model?: Scored; new_network?: Scored; comskip?: Scored; log_loss?: number; auc?: number; how: string; stale: boolean };
export type Info = {
  number?: number;
  /** adbreak-0.<number>-<trees>. */
  name?: string;
  /** What changed in it. */
  note?: string;
  /** Its scores when it was trained (scores.json). */
  scores?: { recordings: number; totals: Total[] };
  version: string;
  exported_at: string;
  trained_on: string[];
  seconds: number;
  break_seconds: number;
  kind_seconds: Record<string, number>;
  trees: number;
  max_trees: number;
  loss: { train?: number[]; validation?: number[] };
};
export type Coverage = { channel: string; network: string; recordings: number; seconds: number; breaks: number; double_box_seconds: number; ad_read_seconds: number; new_network_f1: number | null };
/** A training run (the app's or the CLI's). */
export type Run = {
  id: number; kind: string; origin: string; note: string; started_at: string; finished_at: string | null;
  status: "running" | "ok" | "failed" | "interrupted"; progress: string; error: string | null; recordings: number | null;
  model_name: string | null; model_version: string | null; scores: { totals: Total[] } | null;
};
export type Overview = {
  dir: string; info: Info | null; model_bytes: number; image_model_bytes: number; image_model: string; kinds: string[];
  learning_curve: Curve | null; totals: Total[]; recordings: RecScore[]; can_run: boolean; history: Info[]; coverage: Coverage[]; runs: Run[];
};
export type Job = { running: boolean; action: string; only: string | null; ok: boolean | null; log: string[] };
export type Action = "evaluate" | "retrain" | "learning-curve" | "install";

export const pct = (x: number) => `${Math.round(100 * x)}%`;
export const secs = (x: number) => `${x.toFixed(1)} s`;
export const perHour = (x: number) => `${Math.round(x)} s`;
export const hours = (s: number) => (s >= 3600 ? `${(s / 3600).toFixed(1)} h` : `${Math.round(s / 60)} min`);
export const mb = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} kB`);
/** adbreak-0.2-300; for a model from before names, its number or short hash. */
export const modelName = (i: Info) => i.name ?? (i.number ? `model ${i.number}` : i.version.slice(0, 8));
/** A model's kept commercial-break totals from `source` (model, model-new-network, comskip). */
export const kept = (i: Info | null | undefined, source: string) => i?.scores?.totals.find((t) => t.source === source);
export const RUN_KIND: Record<string, string> = { retrain: "Retrain and install", evaluate: "Rerun on recordings", export: "Export", "learning-curve": "Learning curve", install: "Reinstall a model", refresh: "Refresh all sources" };
export const origin = (r: Run) => (r.origin === "cli" ? "command line" : r.origin === "ingest" ? "an import" : "this app");
export const took = (r: Run) => {
  const s = ((r.finished_at ? new Date(r.finished_at) : new Date()).getTime() - new Date(r.started_at).getTime()) / 1000;
  return s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`;
};
/** "evaluate 7/40: ..." → 7/40. */
export const fraction = (progress: string): [number, number] | null => {
  const m = progress.match(/(\d+)\/(\d+)/);
  return m ? [Number(m[1]), Number(m[2])] : null;
};

/** The model's data; while a run is going (from here or elsewhere), followed every few seconds. */
export function useModel() {
  const [m, setM] = useState<Overview | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const load = useCallback(() => {
    fetch("/api/model")
      .then(async (r) => (r.ok ? r.json() : Promise.reject(new Error((await r.text()) || r.statusText))))
      .then(setM, (e) => setError(e.message));
    fetch("/api/model/run").then((r) => r.json()).then(setJob).catch(() => {});
  }, []);
  useEffect(load, [load]);
  const active = m?.runs.find((r) => r.status === "running");
  const going = !!job?.running || !!active;
  useEffect(() => {
    if (!going) return;
    const t = setInterval(load, 2500);
    return () => clearInterval(t);
  }, [going, load]);
  const run = async (action: Action, opts: { only?: string; version?: string; note?: string } = {}) => {
    const r = await fetch("/api/model/run", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ action, ...opts }) });
    if (!r.ok) setError(await r.text());
    else setError("");
    load();
    return r.ok;
  };
  return { m, job, active, busy: going || !m?.can_run, error, run, reload: load };
}
