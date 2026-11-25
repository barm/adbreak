import type { CSSProperties } from "react";
import { Stack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { List, ListItem } from "@astryxdesign/core/List";

const trained = "var(--color-data-categorical-blue, var(--color-accent))";
const cell: CSSProperties = { width: 14, height: 14, borderRadius: 3 };

/** How the model is trained and scored: the installed model sees every labelled recording; each recording's score
 *  comes from a model that didn't. A grid of which recordings each model was trained on. */
export function HowItWorks({ ids, titles }: { ids: string[]; titles: Record<string, string> }) {
  const n = ids.length;
  const row = (label: string, held: number | null) => (
    <Stack key={label} direction="horizontal" gap={1} align="center">
      <Text type="supporting" color="secondary" style={{ width: 230 }}>{label}</Text>
      {ids.map((id, j) => (
        <Stack
          key={id}
          style={{ ...cell, background: j === held ? "transparent" : trained, border: j === held ? "2px solid var(--color-text-secondary)" : "none" }}
          aria-label={j === held ? `${titles[id]}: held out` : `${titles[id]}: trained on`}
        />
      ))}
    </Stack>
  );
  return (
    <Stack gap={3}>
      <List>
        <ListItem
          label="One installed model, trained on every labelled recording"
          description={`What the DVR runs, and what drafts suggestions on new recordings. It has seen all ${n}, so it can't be scored on them fairly.`}
        />
        <ListItem
          label={`${n} test models, one per recording, each trained on the other ${n - 1}`}
          description="A recording's score (F1, log loss, AUC; the model rows on its labelling page) comes from the test model that left it out: how the model does on a show it has never seen. Fixed: the same split every run, changing only when recordings or labels change."
        />
        <ListItem
          label="Inside each model, the trees also hold out a tenth of their training seconds"
          description="To decide when to stop adding trees (the held-out log loss under Training). A fixed random tenth, seeded, so it's the same every run."
        />
        <ListItem
          label="The learning curve repeats the test with fewer recordings"
          description="For each recording held out, 1, 2, … of the others (up to 3 random sets each, seeded): how much each extra recording helps."
        />
      </List>
      <Stack gap={1}>
        <Text type="label">Which recordings each model trained on</Text>
        <Stack direction="horizontal" gap={2} align="center">
          <Stack direction="horizontal" gap={1} align="center"><Stack style={{ ...cell, background: trained }} /><Text type="supporting" color="secondary">trained on</Text></Stack>
          <Stack direction="horizontal" gap={1} align="center"><Stack style={{ ...cell, border: "2px solid var(--color-text-secondary)" }} /><Text type="supporting" color="secondary">held out (scored)</Text></Stack>
        </Stack>
        {row("Installed model", null)}
        {ids.map((id, i) => row(`Test model for ${titles[id] ?? id}`, i))}
      </Stack>
    </Stack>
  );
}
