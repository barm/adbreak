import type { ReactNode } from "react";
import { Popover } from "@astryxdesign/core/Popover";
import { Stack } from "@astryxdesign/core/Stack";
import { Heading, Text } from "@astryxdesign/core/Text";
import { Divider } from "@astryxdesign/core/Divider";
import { Table } from "@astryxdesign/core/Table";
import { Kbd } from "@astryxdesign/core/Kbd";

export const HELP: [string, string][] = [
  ["Space", "play / pause"],
  ["← →", "1 s (⇧ 10 s, ⌥ 60 s)"],
  [", .", "one frame"],
  ["I O", "mark a new label's start and end here, in either order; with a label selected, move its edge here"],
  ["[ ]", "mark a stretch's start and end here, in either order, for X or ⌫; with one chosen, move that end"],
  ["A", "accept the selected suggestion as a label"],
  ["S", "split the selected label at the playhead: say, an ad read at the end of a break"],
  ["K", "commercial break → double box → ad read → paid programming"],
  ["⌫", "delete the selected label; with a stretch chosen, include it again"],
  ["N P", "next / previous edge from any source"],
  ["drag", "across the bottom row: choose a stretch, as [ ] do"],
  ["X", "exclude the chosen stretch: bad quality, never scored or trained on"],
  ["- =", "zoom the timeline out, in"],
  ["⌘Z ⌘⇧Z", "undo, redo"],
  ["\\", "hide or show this panel"],
  ["?", "show these shortcuts"],
  ["Esc", "clear the selection"],
];
/** Every key and gesture, in a popover: press ? (or click its trigger). */
export function Shortcuts({ isOpen, onOpenChange, children }: { isOpen: boolean; onOpenChange: (open: boolean) => void; children: ReactNode }) {
  return (
    <Popover
      label="Keyboard shortcuts"
      placement="below"
      alignment="end"
      width={520}
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      content={
        <Stack gap={2}>
          <Heading level={4}>Keyboard shortcuts</Heading>
          <Divider />
          <Table
            data={HELP.map(([keys, what]) => ({ keys, what }))}
            idKey="keys"
            density="compact"
            dividers="none"
            columns={[
              { key: "keys", header: "Key", width: { type: "pixel", value: 140 }, renderCell: (h: { keys: string }) => <Kbd keys={h.keys} /> },
              { key: "what", header: "Does" },
            ]}
          />
          <Text type="supporting" color="secondary">
          Click a label's start or end to close in on that edge, its length to frame it. Drag anywhere on the timeline to scrub; click a span
          to select it; drag a label's start or end to move it (where two labels touch, both move; hold ⌥ to move one). Pinch, or ⌥-scroll,
          on the timeline zooms around the pointer, anywhere else around the playhead; swipe sideways to pan. Right-click a label for its
          kind, split and delete; the reviewed row to exclude bad stretches. Changes save as you go.
        </Text>
        </Stack>
      }>
      {children}
    </Popover>
  );
}
