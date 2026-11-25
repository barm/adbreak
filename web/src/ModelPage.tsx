import { useCallback, useEffect, useState } from "react";
import { Layout, LayoutContent } from "@astryxdesign/core/Layout";
import { Stack } from "@astryxdesign/core/Stack";
import { Section } from "@astryxdesign/core/Section";
import { Heading, Text } from "@astryxdesign/core/Text";
import { Table } from "@astryxdesign/core/Table";
import { Link } from "@astryxdesign/core/Link";
import { Button } from "@astryxdesign/core/Button";
import { Badge } from "@astryxdesign/core/Badge";
import { CodeBlock } from "@astryxdesign/core/CodeBlock";
import { MetadataList, MetadataListItem } from "@astryxdesign/core/MetadataList";
import { AppHeader } from "./AppHeader";
import { LearningCurve, type Curve } from "./LearningCurve";
import { HowItWorks } from "./HowItWorks";
import { TrainingRole } from "./TrainingRole";
import { RowLinks } from "./RowLinks";

type Scored = { precision: number; recall: number; f1: number; breaks: number; found: number; matched: number; start_error: number; end_error: number ; wrongly_skipped_per_hour: number; missed_per_hour: number };
type Total = Scored & { source: string; kind: string };
type Rec = { id: string; title: string; model?: Scored; new_network?: Scored; comskip?: Scored; log_loss?: number; auc?: number; how: string; stale: boolean };
type Info = {
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
type Overview = {
  dir: string;
  info: Info | null;
  model_bytes: number;
  image_model_bytes: number;
  image_model: string;
  kinds: string[];
  learning_curve: Curve | null;
  totals: Total[];
  recordings: Rec[];
  can_run: boolean;
};
type Job = { running: boolean; action: string; only: string | null; ok: boolean | null; log: string[] };

const pct = (x: number) => `${Math.round(100 * x)}%`;
const mb = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} kB`);
const hours = (s: number) => (s >= 3600 ? `${(s / 3600).toFixed(1)} h` : `${Math.round(s / 60)} min`);
const secs = (x: number) => `${x.toFixed(1)} s`;

/** The installed model: what it is, how training went, how it scores, and running it again. */
export function ModelPage() {
  const [m, setM] = useState<Overview | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const load = useCallback(() => {
    fetch("/api/model")
      .then(async (r) => (r.ok ? r.json() : Promise.reject(new Error((await r.text()) || r.statusText))))
      .then(setM, (e) => setError(e.message));
  }, []);
  const poll = useCallback(() => fetch("/api/model/run").then((r) => r.json()).then(setJob).catch(() => {}), []);
  useEffect(() => { load(); poll(); }, [load, poll]);
  // While a run is going, follow it; when it ends, reload the scores.
  useEffect(() => {
    if (!job?.running) return;
    const t = setInterval(async () => {
      const j: Job = await fetch("/api/model/run").then((r) => r.json());
      setJob(j);
      if (!j.running) load();
    }, 2000);
    return () => clearInterval(t);
  }, [job?.running, load]);
  const run = async (action: "evaluate" | "retrain" | "learning-curve", only?: string) => {
    const r = await fetch("/api/model/run", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ action, only }) });
    if (!r.ok) setError(await r.text());
    else { setError(""); setJob(await r.json()); }
  };
  const busy = !!job?.running || !m?.can_run;
  const info = m?.info;
  const loss = info?.loss;
  const last = (v?: number[]) => (v && v.length ? v[v.length - 1] : undefined);
  const best = (v?: number[]) => (v && v.length ? Math.min(...v) : undefined);

  return (
    <Layout
      header={
        <AppHeader
          section="model"
          end={
            <>
              <Button label="Rerun on all recordings" size="sm" variant="secondary" isDisabled={busy} tooltip="Each recording's suggestions again, leave-one-out" onClick={() => run("evaluate")} />
              <Button label="Retrain and install" size="sm" variant="primary" isDisabled={busy} tooltip="Rerun on all, then export the model the DVR loads" onClick={() => run("retrain")} />
            </>
          }
        />
      }
      content={
        <LayoutContent padding={3}>
          <Stack gap={4}>
            {error && <Text color="accent">{error}</Text>}
            {m && !m.can_run && <Text color="secondary">Actions are off: no ML project here (adbreak label --ml).</Text>}
            {job && (job.running || job.log.length > 0) && (
              <Section padding={3} variant="muted">
                <Stack gap={2}>
                  <Stack direction="horizontal" gap={2} align="center">
                    <Badge label={job.running ? "running" : job.ok ? "done" : "failed"} variant={job.running ? "info" : job.ok ? "success" : "error"} />
                    <Text type="label">{job.action === "retrain" ? "Retrain and install" : job.action === "learning-curve" ? "Learning curve" : job.action === "ingest" ? `Importing ${job.only}` : job.only ? `Rerun on ${job.only}` : "Rerun on all recordings"}</Text>
                  </Stack>
                  <CodeBlock code={job.log.slice(-12).join("\n") || "…"} size="sm" hasCopyButton={false} />
                </Stack>
              </Section>
            )}
            {m && (
              <Stack direction="horizontal" gap={4} wrap="wrap" align="start">
                <Section padding={3}>
                  <Stack gap={2}>
                    <Heading level={3}>Installed model</Heading>
                    <MetadataList>
                      <MetadataListItem label="Version">{info?.version ?? "–"}</MetadataListItem>
                      <MetadataListItem label="Exported">{info ? new Date(info.exported_at).toLocaleString() : "before this was recorded"}</MetadataListItem>
                      <MetadataListItem label="Size">{`${mb(m.model_bytes)} model + ${mb(m.image_model_bytes)} image model (${m.image_model})`}</MetadataListItem>
                      <MetadataListItem label="Outputs">{["commercial break", ...m.kinds.map((k) => k.replace("_", " "))].join(", ")}</MetadataListItem>
                      <MetadataListItem label="Trees">{info ? `${info.trees} of up to ${info.max_trees}` : "–"}</MetadataListItem>
                      <MetadataListItem label="Directory">{m.dir}</MetadataListItem>
                    </MetadataList>
                  </Stack>
                </Section>
                <Section padding={3}>
                  <Stack gap={2}>
                    <Heading level={3}>Training</Heading>
                    {info ? (
                      <MetadataList>
                        <MetadataListItem label="Recordings">{`${info.trained_on.length}`}</MetadataListItem>
                        <MetadataListItem label="Seconds">{`${hours(info.seconds)} labelled, ${hours(info.break_seconds)} of it commercial breaks`}</MetadataListItem>
                        <MetadataListItem label="Other kinds">{Object.entries(info.kind_seconds).map(([k, s]) => `${k.replace("_", " ")} ${Math.round(s / 60)} min`).join(", ")}</MetadataListItem>
                        <MetadataListItem label="Log loss">
                          {loss?.train
                            ? `training ${last(loss.train)!.toFixed(3)}, held-out ${last(loss.validation)!.toFixed(3)} (best ${best(loss.validation)!.toFixed(3)}; from ${loss.train[0].toFixed(3)} before the first tree)`
                            : "not recorded"}
                        </MetadataListItem>
                      </MetadataList>
                    ) : (
                      <Text color="secondary">Retrain to record it.</Text>
                    )}
                  </Stack>
                </Section>
              </Stack>
            )}
            {m && (
              <Stack gap={2}>
                <Heading level={3}>How this works</Heading>
                <HowItWorks ids={m.recordings.map((r) => r.id)} titles={Object.fromEntries(m.recordings.map((r) => [r.id, r.title]))} />
              </Stack>
            )}
            {m && (
              <Stack gap={2}>
                <Stack direction="horizontal" gap={2} align="center" justify="between">
                  <Heading level={3}>Learning curve</Heading>
                  <Button label={m.learning_curve ? "Recompute" : "Compute"} size="sm" variant="secondary" isDisabled={busy} tooltip="Train on 1, 2, … recordings and score on the rest (a few minutes)" onClick={() => run("learning-curve")} />
                </Stack>
                {m.learning_curve ? (
                  <>
                    <Text type="supporting" color="secondary">
                      {`The commercial-break model trained on k labelled recordings, scored on each one it never saw: ${m.learning_curve.sets} random training sets per k (${m.learning_curve.fits} models in all). Line: median; band: the middle half of scores. Computed ${new Date(m.learning_curve.computed_at).toLocaleString()} over ${m.learning_curve.recordings} recordings.`}
                    </Text>
                    <LearningCurve curve={m.learning_curve} />
                  </>
                ) : (
                  <Text type="supporting" color="secondary">Not computed yet.</Text>
                )}
              </Stack>
            )}
            {m && (
              <Stack gap={2}>
                <Heading level={3}>Scores, leave-one-out</Heading>
                <Text type="supporting" color="secondary">
                  Each recording scored by a model trained on the others, over its reviewed time. The model-new-network row trains only on other networks, which share none of its promos, graphics or ads: the closer test of a channel it hasn't seen. Show cut: programme skipped by mistake; ads left: break time not skipped; both per hour watched.
                </Text>
                <Table
                  data={m.totals.map((t, i) => ({ ...t, i }))}
                  idKey="i"
                  density="compact"
                  columns={[
                    { key: "kind", header: "Kind" },
                    { key: "source", header: "Source" },
                    { key: "f1", header: "F1", align: "end" as const, renderCell: (t: Total) => pct(t.f1) },
                    { key: "precision", header: "Precision", align: "end" as const, renderCell: (t: Total) => pct(t.precision) },
                    { key: "recall", header: "Recall", align: "end" as const, renderCell: (t: Total) => pct(t.recall) },
                    { key: "matched", header: "Found", align: "end" as const, renderCell: (t: Total) => `${t.matched} of ${t.breaks}` },
                    { key: "start_error", header: "Start / end error", align: "end" as const, renderCell: (t: Total) => (t.matched ? `${secs(t.start_error)} / ${secs(t.end_error)}` : "") },
                    { key: "wrongly_skipped_per_hour", header: "Show cut / h", align: "end" as const, renderCell: (t: Total) => (t.kind === "commercial break" ? `${Math.round(t.wrongly_skipped_per_hour)} s` : "") },
                    { key: "missed_per_hour", header: "Ads left / h", align: "end" as const, renderCell: (t: Total) => (t.kind === "commercial break" ? `${Math.round(t.missed_per_hour)} s` : "") },
                  ]}
                />
              </Stack>
            )}
            {m && (
              <Stack gap={2}>
                <Heading level={3}>By recording</Heading>
                <RowLinks ids={m.recordings.map((r) => r.id)}>
                <Table
                  data={m.recordings}
                  idKey="id"
                  density="compact"
                  hasHover
                  textOverflow="truncate"
                  columns={[
                    { key: "title", header: "Recording", width: { type: "proportional" as const, value: 3, minWidth: 240 }, renderCell: (r: Rec) => <Link href={`#/${encodeURIComponent(r.id)}`}>{r.title}</Link> },
                    { key: "f1", header: "Model F1", align: "end" as const, renderCell: (r: Rec) => (r.model ? pct(r.model.f1) : "") },
                    { key: "new_network", header: "New network F1", align: "end" as const, renderCell: (r: Rec) => (r.new_network ? pct(r.new_network.f1) : "") },
                    { key: "found", header: "Found", align: "end" as const, renderCell: (r: Rec) => (r.model ? `${r.model.matched} of ${r.model.breaks}` : "") },
                    { key: "edges", header: "Start / end error", align: "end" as const, renderCell: (r: Rec) => (r.model?.matched ? `${secs(r.model.start_error)} / ${secs(r.model.end_error)}` : "") },
                    { key: "log_loss", header: "Log loss", align: "end" as const, renderCell: (r: Rec) => (r.log_loss != null ? r.log_loss.toFixed(3) : "") },
                    { key: "auc", header: "AUC", align: "end" as const, renderCell: (r: Rec) => (r.auc != null ? r.auc.toFixed(3) : "") },
                    { key: "comskip", header: "Comskip F1", align: "end" as const, renderCell: (r: Rec) => (r.comskip ? pct(r.comskip.f1) : "") },
                    { key: "role", header: "Installed model", renderCell: (r: Rec) => <TrainingRole inTraining={!!info?.trained_on.includes(r.id)} /> },
                    { key: "stale", header: "", renderCell: (r: Rec) => (r.stale ? <Badge label="stale" variant="warning" /> : null) },
                    {
                      key: "run", header: "",
                      renderCell: (r: Rec) => <Button label="Rerun" size="sm" variant="ghost" isDisabled={busy} tooltip={`Rerun the model on ${r.title}`} onClick={() => run("evaluate", r.id)} />,
                    },
                  ]}
                />
                </RowLinks>
              </Stack>
            )}
          </Stack>
        </LayoutContent>
      }
    />
  );
}
