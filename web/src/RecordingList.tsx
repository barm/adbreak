import { useEffect, useState } from "react";
import { Layout, LayoutContent, LayoutHeader } from "@astryxdesign/core/Layout";
import { Table } from "@astryxdesign/core/Table";
import { Heading, Text } from "@astryxdesign/core/Text";
import { Link } from "@astryxdesign/core/Link";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { Stack } from "@astryxdesign/core/Stack";
import { fmt } from "./editor/engine";

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

/** Every imported recording, newest first. */
export function RecordingList() {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    fetch("/api/recordings")
      .then(async (r) => (r.ok ? r.json() : Promise.reject(new Error((await r.text()) || r.statusText))))
      .then(setRows, (e) => setError(e.message));
  }, []);
  const columns = [
    {
      key: "id",
      header: "Recording",
      renderCell: (r: Row) => (r.media ? <Link href={`#/${encodeURIComponent(r.id)}`}>{r.id}</Link> : <Text color="secondary">{r.id}</Text>),
    },
    { key: "title", header: "Title" },
    { key: "channel", header: "Channel", renderCell: (r: Row) => `${r.channel} ${r.network}` },
    { key: "start", header: "Recorded", renderCell: (r: Row) => new Date(r.start).toLocaleString() },
    { key: "duration", header: "Length", align: "end" as const, renderCell: (r: Row) => clock(r.duration) },
    {
      key: "reviewed",
      header: "Reviewed",
      align: "end" as const,
      // Of what there is to review: excluded parts (bad reception) don't count.
      renderCell: (r: Row) => {
        const pct = r.usable ? Math.min(100, Math.round((100 * r.reviewed) / r.usable)) : 0;
        return pct ? `${pct}%` : "";
      },
    },
    { key: "breaks", header: "Labels", align: "end" as const, renderCell: (r: Row) => r.breaks || "" },
    {
      key: "sources",
      header: "Suggestions",
      renderCell: (r: Row) =>
        r.media ? r.sources.join(", ") || `none: run adbreak suggest ${r.id}` : <Text color="secondary">no media: import it again</Text>,
    },
  ];
  return (
    <Layout
      height="auto"
      header={
        <LayoutHeader padding={3}>
          <Heading level={1}>adbreak labelling</Heading>
        </LayoutHeader>
      }
      content={
        <LayoutContent padding={3}>
          <Stack gap={2}>
            {error && <Text color="accent">{error}</Text>}
            {rows && (
              <Table
                data={rows}
                columns={columns}
                idKey="id"
                hasHover
                density="compact"
                emptyState={<EmptyState title="Nothing imported yet" description="adbreak import <dir>" isCompact />}
              />
            )}
          </Stack>
        </LayoutContent>
      }
    />
  );
}
