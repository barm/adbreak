import { useState } from "react";
import { Layout, LayoutContent, VStack, HStack } from "@astryxdesign/core/Layout";
import { Heading, Text } from "@astryxdesign/core/Text";
import { Card } from "@astryxdesign/core/Card";
import { Table } from "@astryxdesign/core/Table";
import { Button } from "@astryxdesign/core/Button";
import { Badge } from "@astryxdesign/core/Badge";
import { TextInput } from "@astryxdesign/core/TextInput";
import { ProgressBar } from "@astryxdesign/core/ProgressBar";
import { CodeBlock } from "@astryxdesign/core/CodeBlock";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { Link } from "@astryxdesign/core/Link";
import { AppHeader } from "./AppHeader";
import { useModel, RUN_KIND, origin, took, fraction, pct, perHour, type Run } from "./model";

const STATUS: Record<Run["status"], "info" | "success" | "error" | "warning"> = { running: "info", ok: "success", failed: "error", interrupted: "warning" };
const score = (r: Run) => r.scores?.totals.find((t) => t.source === "model");

/** Training runs: start one, follow the one going (from here, the command line or an import), and every past one. */
export function RunsPage() {
  const { m, job, active, busy, error, run } = useModel();
  const [note, setNote] = useState("");
  const retrain = async () => {
    if (await run("retrain", { note })) setNote("");
  };
  const f = active ? fraction(active.progress) : null;
  // The app's own log, while its run is the one going (a command-line run's log is in its terminal).
  const log = job && active?.origin === "app" ? job.log : null;
  return (
    <Layout
      header={<AppHeader section="runs" />}
      content={
        <LayoutContent padding={4}>
          <VStack gap={6}>
            {error && <Text color="accent">{error}</Text>}
            {m && !m.can_run && <Text color="secondary">Runs can't start from here: no ML project (adbreak label --ml).</Text>}
            <Card>
              <VStack gap={3}>
                <Heading level={4}>Start a run</Heading>
                <HStack gap={2} vAlign="end" wrap="wrap">
                  <TextInput label="What changed" placeholder="optional; kept with the model" value={note} onChange={setNote} />
                  <Button label="Retrain and install" variant="primary" isDisabled={busy} tooltip="Score every recording leave-one-out, then train on all and install the next model; the DVR picks it up" onClick={retrain} />
                  <Button label="Rerun on all recordings" variant="secondary" isDisabled={busy} tooltip="Each recording's suggestions and scores again, without installing anything" onClick={() => run("evaluate")} />
                </HStack>
                <Text type="supporting" color="secondary">One run at a time: these are off while one is going, wherever it started. A retrain takes about an hour with today's recordings.</Text>
              </VStack>
            </Card>
            {active && (
              <Card>
                <VStack gap={3}>
                  <HStack gap={2} vAlign="center" wrap="wrap">
                    <Badge label="running" variant="info" />
                    <Heading level={4}>{RUN_KIND[active.kind] ?? active.kind}</Heading>
                    <Text color="secondary">{`from ${origin(active)} · ${took(active)}`}</Text>
                  </HStack>
                  {active.note && <Text>{active.note}</Text>}
                  {f ? (
                    <ProgressBar label="Progress" value={f[0]} max={f[1]} hasValueLabel formatValueLabel={(v, mx) => `${v} of ${mx}`} />
                  ) : null}
                  <Text type="supporting" color="secondary">{active.progress || "starting"}</Text>
                  {log && log.length > 0 && <CodeBlock code={log.slice(-10).join("\n")} size="sm" hasCopyButton={false} />}
                </VStack>
              </Card>
            )}
            {m && (
              <VStack gap={3}>
                <Heading level={3}>History</Heading>
                <Table
                  data={m.runs}
                  idKey="id"
                  density="compact"
                  dividers="rows"
                  emptyState={<EmptyState title="No runs yet" description="Runs from this page and from adbreak-ml on the command line show here." isCompact />}
                  columns={[
                    { key: "started_at", header: "Started", renderCell: (r: Run) => new Date(r.started_at).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) },
                    { key: "kind", header: "Run", renderCell: (r: Run) => <VStack gap={0}><Text>{RUN_KIND[r.kind] ?? r.kind}</Text>{r.note && <Text type="supporting" color="secondary">{r.note}</Text>}</VStack> },
                    { key: "origin", header: "From", renderCell: origin },
                    { key: "took", header: "Took", align: "end" as const, renderCell: took },
                    { key: "status", header: "", renderCell: (r: Run) => <Badge label={r.status} variant={STATUS[r.status]} /> },
                    { key: "detail", header: "", renderCell: (r: Run) => <Text type="supporting" color="secondary">{r.status === "running" ? r.progress : r.status === "failed" ? r.error ?? "" : r.recordings ? `${r.recordings} recordings` : ""}</Text> },
                    { key: "model", header: "Model made", renderCell: (r: Run) => (r.model_name ? <Link href="#/models">{r.model_name}</Link> : "") },
                    { key: "f1", header: "F1", align: "end" as const, renderCell: (r: Run) => (score(r) ? pct(score(r)!.f1) : "") },
                    { key: "cut", header: "Show cut / h", align: "end" as const, renderCell: (r: Run) => (score(r) ? perHour(score(r)!.wrongly_skipped_per_hour) : "") },
                    { key: "left", header: "Ads left / h", align: "end" as const, renderCell: (r: Run) => (score(r) ? perHour(score(r)!.missed_per_hour) : "") },
                  ]}
                />
              </VStack>
            )}
          </VStack>
        </LayoutContent>
      }
    />
  );
}
