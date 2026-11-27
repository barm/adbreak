import { useEffect, useMemo, useState } from "react";
import { Layout, LayoutContent, VStack, HStack } from "@astryxdesign/core/Layout";
import { Table } from "@astryxdesign/core/Table";
import { Text } from "@astryxdesign/core/Text";
import { Link } from "@astryxdesign/core/Link";
import { Button } from "@astryxdesign/core/Button";
import { Badge } from "@astryxdesign/core/Badge";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import { fmt } from "./editor/engine";
import { AppHeader } from "./AppHeader";
import { TrainingRole } from "./TrainingRole";
import { RowLinks } from "./RowLinks";
import { useModel, pct, secs, perHour, hours, modelName, type Coverage, type RecScore } from "./model";

type Row = {
  id: string;
  title: string;
  channel: string;
  network: string;
  start: string;
  duration: number;
  reviewed: number;
  usable: number;
  breaks: number;
  sources: string[];
  media: boolean;
};

const clock = (t: number) => fmt(t).replace(/\.\d$/, "");
const ready = (r: Row) => r.reviewed >= r.duration - 1;
const status = (r: Row) => (ready(r) ? "ready" : r.reviewed > 0 ? `${pct(Math.min(1, r.reviewed / r.duration))} reviewed` : r.breaks > 0 ? "labelling" : "not started");
const when = (s: string) => new Date(s).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

/** Every imported recording, newest first, with how the model does on each; or the same by channel. */
export function RecordingList() {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [error, setError] = useState("");
  const [view, setView] = useState<"recordings" | "channels">("recordings");
  const { m, busy, run } = useModel();
  useEffect(() => {
    fetch("/api/recordings")
      .then(async (r) => (r.ok ? r.json() : Promise.reject(new Error((await r.text()) || r.statusText))))
      .then(setRows, (e) => setError(e.message));
  }, []);
  const scores = useMemo(() => new Map((m?.recordings ?? []).map((r) => [r.id, r])), [m]);
  const trainedOn = useMemo(() => new Set(m?.info?.trained_on ?? []), [m]);
  const held = useMemo(() => new Set(m?.holdout ?? []), [m]);
  const s = (r: Row): RecScore | undefined => scores.get(r.id);
  const now = m?.totals.find((t) => t.kind === "commercial break" && t.source === "model");
  const labelled = rows?.filter((r) => r.reviewed > 0) ?? [];
  const readyCount = rows?.filter(ready).length ?? 0;

  const columns = [
    {
      key: "title",
      header: "Recording",
      width: { type: "proportional" as const, value: 3, minWidth: 260 },
      renderCell: (r: Row) => (
        <VStack gap={0}>
          {r.media ? <Link href={`#/${encodeURIComponent(r.id)}`}>{r.title}</Link> : <Text color="secondary">{`${r.title} (no media)`}</Text>}
          <Text type="supporting" color="secondary">{`${r.channel} ${r.network} · ${when(r.start)} · ${clock(r.duration)}`}</Text>
        </VStack>
      ),
    },
    // Ready for training (the switch on its page), partly reviewed (from before the switch), being labelled, or not yet.
    { key: "status", header: "Status", renderCell: (r: Row) => <Text color={ready(r) ? "primary" : "secondary"}>{status(r)}</Text> },
    { key: "breaks", header: "Labels", align: "end" as const, renderCell: (r: Row) => r.breaks || "" },
    { key: "f1", header: "F1", align: "end" as const, renderCell: (r: Row) => (s(r)?.model ? pct(s(r)!.model!.f1) : "") },
    { key: "nn", header: "New network", align: "end" as const, renderCell: (r: Row) => (s(r)?.new_network ? pct(s(r)!.new_network!.f1) : "") },
    { key: "found", header: "Found", align: "end" as const, renderCell: (r: Row) => (s(r)?.model ? `${s(r)!.model!.matched} of ${s(r)!.model!.breaks}` : "") },
    { key: "edges", header: "Start / end", align: "end" as const, renderCell: (r: Row) => (s(r)?.model?.matched ? `${secs(s(r)!.model!.start_error)} / ${secs(s(r)!.model!.end_error)}` : "") },
    { key: "cut", header: "Show cut / h", align: "end" as const, renderCell: (r: Row) => (s(r)?.model ? perHour(s(r)!.model!.wrongly_skipped_per_hour) : "") },
    { key: "left", header: "Ads left / h", align: "end" as const, renderCell: (r: Row) => (s(r)?.model ? perHour(s(r)!.model!.missed_per_hour) : "") },
    { key: "comskip", header: "Comskip", align: "end" as const, renderCell: (r: Row) => (s(r)?.comskip ? pct(s(r)!.comskip!.f1) : "") },
    {
      key: "role",
      header: "Model",
      width: { type: "pixel" as const, value: 150 },
      renderCell: (r: Row) =>
        m?.info ? (
          <HStack gap={1}>
            <TrainingRole role={held.has(r.id) ? "held out" : trainedOn.has(r.id) ? "training" : "not trained on"} />
            {s(r)?.stale && <Badge label="stale" variant="warning" />}
          </HStack>
        ) : null,
    },
    { key: "rerun", header: "", renderCell: (r: Row) => (r.media ? <Button label="Rerun" size="sm" variant="ghost" isDisabled={busy} tooltip={`Run the model on ${r.title} again (a short run)`} onClick={() => run("evaluate", { only: r.id })} /> : null) },
  ];
  const channelColumns = [
    { key: "channel", header: "Channel", renderCell: (c: Coverage) => `${c.channel} ${c.network}` },
    { key: "recordings", header: "Recordings", align: "end" as const },
    { key: "seconds", header: "Reviewed", align: "end" as const, renderCell: (c: Coverage) => hours(c.seconds) },
    { key: "breaks", header: "Breaks", align: "end" as const },
    { key: "double_box_seconds", header: "Double box", align: "end" as const, renderCell: (c: Coverage) => (c.double_box_seconds ? hours(c.double_box_seconds) : "") },
    { key: "ad_read_seconds", header: "Ad read", align: "end" as const, renderCell: (c: Coverage) => (c.ad_read_seconds ? hours(c.ad_read_seconds) : "") },
    { key: "new_network_f1", header: "New network F1", align: "end" as const, renderCell: (c: Coverage) => (c.new_network_f1 != null ? pct(c.new_network_f1) : "") },
  ];
  return (
    <Layout
      header={<AppHeader section="recordings" />}
      content={
        <LayoutContent padding={2}>
          <VStack gap={2}>
            {error && <Text color="accent">{error}</Text>}
            <HStack gap={3} vAlign="center" hAlign="between" wrap="wrap">
              <Text color="secondary">
                {rows && `${rows.length} recordings · ${readyCount} ready for training (${hours(labelled.reduce((a, r) => a + r.reviewed, 0))}) · ${labelled.reduce((a, r) => a + r.breaks, 0)} labels`}
                {m?.info && now && ` · ${modelName(m.info)}: F1 ${pct(now.f1)}, show cut ${perHour(now.wrongly_skipped_per_hour)} / h, ads left ${perHour(now.missed_per_hour)} / h${m.holdout.length ? ` on the ${m.holdout.length} held out` : ""}`}
              </Text>
              <SegmentedControl label="View" value={view} onChange={(v) => setView(v as "recordings" | "channels")} size="sm">
                <SegmentedControlItem value="recordings" label="Recordings" />
                <SegmentedControlItem value="channels" label="By channel" />
              </SegmentedControl>
            </HStack>
            {view === "recordings" && rows && (
              <RowLinks ids={rows.map((r) => (r.media ? r.id : ""))}>
                <Table
                  data={rows}
                  columns={columns}
                  idKey="id"
                  hasHover
                  density="compact"
                  textOverflow="truncate"
                  emptyState={<EmptyState title="Nothing imported yet" description="adbreak import <dir>, or mark a recording for Adbreak in the recorder" isCompact />}
                />
              </RowLinks>
            )}
            {view === "channels" && m && (
              <VStack gap={2}>
                <Text type="supporting" color="secondary">
                  What the labelled data holds per channel, least first. More recordings teach the model most where there is little, where its score on an unseen network is weak, and in programmes with double boxes or ad reads.
                </Text>
                <Table data={m.coverage} idKey="channel" density="compact" columns={channelColumns} />
              </VStack>
            )}
          </VStack>
        </LayoutContent>
      }
    />
  );
}
