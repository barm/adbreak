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
  ["1 – 6", "set the selected label's kind: 1 commercial break, 2 double box, 3 ad read, 4 bumper, 5 credits, 6 paid programming"],
  ["K", "cycle the selected label's kind through those"],
  ["⌫", "delete the selected label; with a stretch chosen, include it again"],
  ["↑ ↓", "previous / next edit: a cut, an edge, a label's or a source's edge"],
  ["N", "snapping on or off: moved times go to the nearest cut, edge or label edge"],
  ["drag", "on empty space in the labels row: a new label; along the time ruler: choose a stretch, as [ ] do; a label's edge: move it (paused while you drag)"],
  ["X", "exclude the chosen stretch: bad quality, never scored or trained on"],
  ["- =", "zoom the timeline out, in"],
  ["⌘Z ⌘⇧Z", "undo, redo"],
  ["\\", "hide or show this panel"],
  ["?", "show these shortcuts"],
  ["Esc", "cancel a stretch or a half-marked label; otherwise stop playback (and clear the selection)"],
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
          kind, split and delete; the time ruler to exclude bad stretches. Changes save as you go.
        </Text>
        </Stack>
      }>
      {children}
    </Popover>
  );
}
