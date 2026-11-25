import { useEffect, useState } from "react";
import { Layout, LayoutContent } from "@astryxdesign/core/Layout";
import { Table } from "@astryxdesign/core/Table";
import { Text } from "@astryxdesign/core/Text";
import { Link } from "@astryxdesign/core/Link";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { Stack } from "@astryxdesign/core/Stack";
import { fmt } from "./editor/engine";
import { AppHeader } from "./AppHeader";

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
type Score = { id: string; model?: { f1: number } };

const clock = (t: number) => fmt(t).replace(/\.\d$/, "");
const when = (s: string) => new Date(s).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const pct = (x: number) => `${Math.round(100 * x)}%`;

/** Every imported recording, newest first. */
export function RecordingList() {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [f1, setF1] = useState<Record<string, number>>({});
  const [error, setError] = useState("");
  useEffect(() => {
    fetch("/api/recordings")
      .then(async (r) => (r.ok ? r.json() : Promise.reject(new Error((await r.text()) || r.statusText))))
      .then(setRows, (e) => setError(e.message));
    fetch("/api/model")
      .then((r) => (r.ok ? r.json() : null))
      .then((m) => m && setF1(Object.fromEntries((m.recordings as Score[]).filter((r) => r.model).map((r) => [r.id, r.model!.f1]))))
      .catch(() => {});
  }, []);
  const columns = [
    {
      key: "title",
      header: "Recording",
      width: { type: "proportional" as const, value: 3, minWidth: 260 },
      renderCell: (r: Row) =>
        r.media ? <Link href={`#/${encodeURIComponent(r.id)}`}>{r.title}</Link> : <Text color="secondary">{r.title} (no media)</Text>,
    },
    { key: "channel", header: "Channel", renderCell: (r: Row) => `${r.channel} ${r.network}` },
    { key: "start", header: "Recorded", renderCell: (r: Row) => when(r.start) },
    { key: "duration", header: "Length", align: "end" as const, renderCell: (r: Row) => clock(r.duration) },
    {
      key: "reviewed",
      header: "Reviewed",
      align: "end" as const,
      // Of what there is to review: excluded parts (bad reception) don't count.
      renderCell: (r: Row) => (r.usable && r.reviewed ? pct(Math.min(1, r.reviewed / r.usable)) : ""),
    },
    { key: "breaks", header: "Labels", align: "end" as const, renderCell: (r: Row) => r.breaks || "" },
    { key: "f1", header: "Model F1", align: "end" as const, renderCell: (r: Row) => (r.id in f1 ? pct(f1[r.id]) : "") },
    { key: "id", header: "ID", renderCell: (r: Row) => <Text type="supporting" color="secondary">{r.id}</Text> },
  ];
  return (
    <Layout
      header={<AppHeader section="recordings" />}
      content={
        <LayoutContent padding={2}>
          <Stack gap={2}>
            {error && <Text color="accent">{error}</Text>}
            {rows && (
              <Table
                data={rows}
                columns={columns}
                idKey="id"
                hasHover
                density="compact"
                textOverflow="truncate"
                emptyState={<EmptyState title="Nothing imported yet" description="adbreak import <dir>" isCompact />}
              />
            )}
          </Stack>
        </LayoutContent>
      }
    />
  );
}
