import { Toolbar } from "@astryxdesign/core/Toolbar";
import { Text } from "@astryxdesign/core/Text";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import type { ReactNode } from "react";

/** The app's header on its top-level pages: the name, and Recordings | Model. */
export function AppHeader({ section, end }: { section: "recordings" | "model"; end?: ReactNode }) {
  return (
    <Toolbar
      label="Adbreak"
      size="sm"
      dividers={["bottom"]}
      startContent={<Text type="label">Adbreak</Text>}
      centerContent={
        <SegmentedControl label="Section" value={section} onChange={(v) => { location.hash = v === "model" ? "#/model" : ""; }}>
          <SegmentedControlItem value="recordings" label="Recordings" />
          <SegmentedControlItem value="model" label="Model" />
        </SegmentedControl>
      }
      endContent={end}
    />
  );
}
