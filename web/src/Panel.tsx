import { useEffect, useState } from "react";
import { Stack } from "@astryxdesign/core/Stack";
import { Table } from "@astryxdesign/core/Table";
import { Text } from "@astryxdesign/core/Text";
import { Button } from "@astryxdesign/core/Button";
import { Selector } from "@astryxdesign/core/Selector";
import { TextArea } from "@astryxdesign/core/TextArea";
import { KINDS, KIND_NAME, fmt } from "./editor/engine";
import type { Engine } from "./Review";

type Label = { start: number; end: number; kind?: string; i: number };
const clock = (t: number) => fmt(t).replace(/\.\d$/, "");
// An edge's time, short: hours only when there are some.
const short = (t: number) => fmt(t).replace(/^0:/, "");


/** The labels as a table, the notes and the help. */
export function Panel({ engine }: { engine: Engine }) {
  const S = engine.state();
  const sel = engine.selection();
  const [notes, setNotes] = useState(S.labels.notes || "");
  useEffect(() => setNotes(S.labels.notes || ""), [S.id]);
  const data: Label[] = S.labels.breaks.map((b: Label, i: number) => ({ ...b, i }));
  const reviewed = S.labels.reviewed.reduce((a: number, r: Label) => a + r.end - r.start, 0);
  const excluded = (S.labels.excluded || []).reduce((a: number, r: Label) => a + r.end - r.start, 0);
  const edge = (b: Label, e: "start" | "end") => (
    <Button label={`Jump to the ${e}`} size="sm" variant="ghost" onClick={() => engine.select(b.i, e)}>
      {short(b[e])}
    </Button>
  );
  return (
    <Stack gap={3}>
      <Stack gap={1}>
        <Text type="supporting" color="secondary">
          {S.labels.breaks.length} labels · reviewed {clock(reviewed)} of {clock(S.dur - excluded)}
          {excluded ? ` (${clock(excluded)} excluded)` : ""}
        </Text>
        <Table
          data={data}
          idKey="i"
          density="compact"
          hasHover
          columns={[
            { key: "start", header: "Start", renderCell: (b: Label) => edge(b, "start") },
            { key: "end", header: "End", renderCell: (b: Label) => edge(b, "end") },
            {
              key: "len", header: "Length", align: "end" as const,
              renderCell: (b: Label) => (
                <Button label="Frame it on the timeline" size="sm" variant={sel && sel.label && sel.i === b.i ? "secondary" : "ghost"} onClick={() => engine.select(b.i, undefined)}>
                  {`${Math.round(b.end - b.start)} s`}
                </Button>
              ),
            },
            {
              key: "kind", header: "Kind",
              renderCell: (b: Label) => (
                <Selector label="Kind" isLabelHidden size="sm" width={150} value={b.kind || "ad"} options={KINDS.map((k) => ({ value: k, label: KIND_NAME[k as keyof typeof KIND_NAME] }))} onChange={(k) => k && engine.setKind(b.i, k)} />
              ),
            },
            { key: "del", header: "", renderCell: (b: Label) => <Button label="Delete" isIconOnly icon={<Text>✕</Text>} size="sm" variant="ghost" tooltip="Delete" onClick={() => engine.deleteBreak(b.i)} /> },
          ]}
        />
      </Stack>
      <TextArea label="Notes" value={notes} rows={2} onChange={(v: string) => { setNotes(v); engine.setNotes(v); }} />
    </Stack>
  );
}
