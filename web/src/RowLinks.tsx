import type { ReactNode, MouseEvent } from "react";
import { Stack } from "@astryxdesign/core/Stack";

/** Makes a whole table row open its recording: a click anywhere on a body row (other than on a control) opens it.
 *  Each row's main cell keeps its Link, for the keyboard and screen readers (Astryx's Table has no row-click). */
export function RowLinks({ ids, children }: { ids: string[]; children: ReactNode }) {
  const open = (e: MouseEvent) => {
    const el = e.target as HTMLElement;
    if (el.closest("a, button, input, select, textarea, [role=button], [role=menu]")) return;
    const tr = el.closest("tbody tr") as HTMLTableRowElement | null;
    if (!tr || !tr.parentElement) return;
    const i = Array.prototype.indexOf.call(tr.parentElement.children, tr);
    if (ids[i]) location.hash = `#/${encodeURIComponent(ids[i])}`;
  };
  return <Stack onClick={open} style={{ cursor: "pointer" }}>{children}</Stack>;
}
