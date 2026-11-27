import { Toolbar } from "@astryxdesign/core/Toolbar";
import { Text } from "@astryxdesign/core/Text";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import type { ReactNode } from "react";

export type Section = "recordings" | "models" | "runs";

/** The app's header on its top-level pages: the name, and Recordings | Models | Runs. */
export function AppHeader({ section, end }: { section: Section; end?: ReactNode }) {
  return (
    <Toolbar
      label="Adbreak"
      size="sm"
      dividers={["bottom"]}
      startContent={<Text type="label">Adbreak</Text>}
      centerContent={
        <SegmentedControl label="Section" value={section} onChange={(v) => { location.hash = v === "recordings" ? "" : `#/${v}`; }}>
          <SegmentedControlItem value="recordings" label="Recordings" />
          <SegmentedControlItem value="models" label="Models" />
          <SegmentedControlItem value="runs" label="Runs" />
        </SegmentedControl>
      }
      endContent={end}
    />
  );
}
