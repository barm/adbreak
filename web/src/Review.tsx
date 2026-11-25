import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { flushSync } from "react-dom";
import { Layout, LayoutContent, LayoutHeader, LayoutPanel } from "@astryxdesign/core/Layout";
import { ResizeHandle, useResizable } from "@astryxdesign/core/Resizable";
import { Stack, StackItem } from "@astryxdesign/core/Stack";
import { Text, Heading } from "@astryxdesign/core/Text";
import { Link } from "@astryxdesign/core/Link";
import { Button } from "@astryxdesign/core/Button";
import { Kbd } from "@astryxdesign/core/Kbd";
import { ContextMenu } from "@astryxdesign/core/ContextMenu";
import { DropdownMenu } from "@astryxdesign/core/DropdownMenu";
import { Popover } from "@astryxdesign/core/Popover";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import { createEngine } from "./editor/engine";
import { Panel } from "./Panel";
import { Shortcuts } from "./Shortcuts";

export type Engine = ReturnType<typeof createEngine>;
type Placement = "below" | "side" | "hidden";
interface Tag { left: number; top: number; text: string; i: number }

// The timeline's name column, as the engine draws it (its NAME).
const NAME = 136;

const styles: Record<string, CSSProperties> = {
  stage: { position: "relative", background: "var(--color-background-inverse, black)", borderRadius: "var(--radius-2)" },
  video: { display: "block", width: "100%", maxHeight: "50vh", background: "black" },
  caption: {
    position: "absolute", insetInline: 0, bottom: "var(--spacing-8)", textAlign: "center", pointerEvents: "none",
    color: "white", textShadow: "0 1px 3px black", whiteSpace: "pre-line",
  },
  timeline: { position: "relative" },
  canvas: { display: "block", width: "100%", cursor: "crosshair", touchAction: "none" },
  overlay: { position: "absolute" },
  time: { fontVariantNumeric: "tabular-nums", minWidth: "7.5em" },
  fill: { minHeight: 0, overflow: "auto" },
};


/** One recording: the video, the timeline, the toolbar and the panel of labels, notes and help. */
export function Review({ id }: { id: string }) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const time = useRef<HTMLElement>(null);
  const cap = useRef<HTMLElement>(null);
  const zoom = useRef<HTMLElement>(null);
  const [engine, setEngine] = useState<Engine | null>(null);
  const [, setTick] = useState(0);
  const [status, setStatus] = useState<{ msg: string; err: boolean }>({ msg: "", err: false });
  const [crumb, setCrumb] = useState("");
  const [tag, setTag] = useState<Tag | null>(null);
  const [menu, setMenu] = useState<object[]>([]);
  const [keysOpen, setKeysOpen] = useState(false);
  const [placement, setPlacement] = useState<Placement>(() => (localStorage.getItem("adbreak.layout") as Placement) || "below");
  const placementRef = useRef(placement);
  placementRef.current = placement;

  const side = useResizable({ defaultSize: 480, minSize: 300, maxSize: 900, autoSaveId: "adbreak.side" });
  const below = useResizable({ defaultSize: 320, minSize: 120, maxSize: Infinity, direction: "vertical", autoSaveId: "adbreak.below" });

  const place = (p: Placement) => {
    if (p !== "hidden") localStorage.setItem("adbreak.shownLayout", p);
    localStorage.setItem("adbreak.layout", p);
    setPlacement(p);
  };

  useEffect(() => {
    const e = createEngine({
      canvas: canvas.current!,
      video: video.current!,
      timeEl: time.current,
      capEl: cap.current,
      zoomEl: zoom.current,
      status: (msg: string, err: boolean) => setStatus({ msg, err }),
      render: () => setTick((n) => n + 1),
      toggleLayout: () => place(placementRef.current === "hidden" ? ((localStorage.getItem("adbreak.shownLayout") as Placement) || "below") : "hidden"),
      kindTag: (t: Tag | null) => setTag((old) => (old && t && old.left === t.left && old.top === t.top && old.text === t.text && old.i === t.i) || old === t ? old : t),
      crumb: setCrumb,
      shortcuts: () => setKeysOpen((o) => !o),
      active: () => true,
    });
    setEngine(e);
    e.load(id).catch((err: Error) => setStatus({ msg: err.message, err: true }));
    return () => e.destroy();
  }, [id]);

  // The canvas follows the layout's size.
  useEffect(() => { engine?.draw(); }, [engine, placement, side.size, below.size]);
  useEffect(() => {
    if (!canvas.current || !engine) return;
    const o = new ResizeObserver(() => engine.draw());
    o.observe(canvas.current);
    return () => o.disconnect();
  }, [engine]);

  const rows = engine?.rows() ?? [];
  const tagItems = useMemo(() => (engine && tag ? engine.labelMenu(tag.i) : []), [engine, tag, menu]);

  const panel = engine && engine.state() ? <Panel engine={engine} /> : null;
  const timeline = (
    <ContextMenu items={menu.length ? (menu as never) : [{ label: "…" }]} menuWidth="260px" label="Label">
      <Stack
        style={styles.timeline}
        onContextMenuCapture={(ev: React.MouseEvent) => {
          const items = engine?.menuAt(ev.nativeEvent);
          // Nothing of ours there: let the browser's own menu open.
          if (!items) { ev.stopPropagation(); return; }
          flushSync(() => setMenu(items));
        }}>
        <canvas ref={canvas} style={styles.canvas} />
        {tag && (
          <Stack style={{ ...styles.overlay, left: tag.left, top: tag.top }}>
            <DropdownMenu
              button={{ label: tag.text, size: "sm", variant: "ghost" }}
              items={tagItems as never}
              onOpenChange={(open) => open && setMenu([...menu])}
              menuWidth="260px"
            />
          </Stack>
        )}
        {engine && rows.map((r) => (
          <Stack key={r.name} style={{ ...styles.overlay, left: NAME - 24, top: r.y + r.h / 2 - 11 }}>
            <InfoButton engine={engine} name={r.name} />
          </Stack>
        ))}
      </Stack>
    </ContextMenu>
  );

  return (
    <Layout
      header={
        <LayoutHeader padding={2}>
          <Stack direction="horizontal" gap={3} align="center" justify="between">
            <Stack direction="horizontal" gap={3} align="center">
              <Link href="#">adbreak labelling</Link>
              <Text color="secondary" type="supporting">{crumb}</Text>
              <Text type="supporting" color={status.err ? "accent" : "secondary"}>{status.msg}</Text>
            </Stack>
            <Stack direction="horizontal" gap={2} align="center">
            <Shortcuts isOpen={keysOpen} onOpenChange={setKeysOpen}>
              <Button label="Keyboard shortcuts" size="sm" variant="ghost" tooltip="Keyboard shortcuts (?)">
                <Stack direction="horizontal" gap={1} align="center">Shortcuts<Kbd keys="?" /></Stack>
              </Button>
            </Shortcuts>
            <SegmentedControl label="Panel" value={placement} onChange={(v) => place(v as Placement)} size="sm">
              <SegmentedControlItem label="Panel below" value="below" />
              <SegmentedControlItem label="At the side" value="side" />
              <SegmentedControlItem label="Hidden" value="hidden" />
            </SegmentedControl>
            </Stack>
          </Stack>
        </LayoutHeader>
      }
      content={
        <LayoutContent>
          <Stack direction="vertical" height="100%">
            <StackItem size="fill" style={styles.fill}>
              <Stack gap={2} padding={2}>
                <Stack style={styles.stage}>
                  <video ref={video} style={styles.video} playsInline controls preload="auto" />
                  <Text ref={cap as never} style={styles.caption} size="lg">{""}</Text>
                </Stack>
                <Stack direction="horizontal" gap={3} align="center">
                  <Text ref={time as never} type="code" style={styles.time}>0:00:00.0</Text>
                  <Text ref={zoom as never} type="supporting" color="secondary">{""}</Text>
                </Stack>
                {timeline}
              </Stack>
            </StackItem>
            {placement === "below" && panel && (
              <>
                <ResizeHandle direction="vertical" hasDivider isReversed resizable={below.props} label="Resize the panel" />
                <Stack height={below.size} style={styles.fill} padding={2}>{panel}</Stack>
              </>
            )}
          </Stack>
        </LayoutContent>
      }
      end={
        placement === "side" && panel ? (
          <>
            <ResizeHandle direction="horizontal" hasDivider isReversed resizable={side.props} label="Resize the panel" />
            <LayoutPanel width={side.size} padding={2}>{panel}</LayoutPanel>
          </>
        ) : undefined
      }
    />
  );
}

/** A timeline row's ⓘ: what the row is, and for a source its score on this recording. */
function InfoButton({ engine, name }: { engine: Engine; name: string }) {
  const [open, setOpen] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const info: any = open ? engine.info(name) : null;
  const pct = (x: number) => `${Math.round(100 * x)}%`;
  return (
    <Popover
      label={name}
      isOpen={open}
      onOpenChange={setOpen}
      placement="end"
      width={360}
      content={
        info && (
          <Stack gap={1}>
            <Heading level={4}>{info.title}</Heading>
            <Text type="supporting">{info.text}</Text>
            {info.warning && <Text type="supporting" color="accent">⚠︎ {info.warning}</Text>}
            {info.details.map((d: string) => <Text key={d} type="supporting" color="secondary">{d}</Text>)}
            {info.score && (
              <Text type="supporting" color="secondary">
                On this recording: precision {pct(info.score.precision)}, recall {pct(info.score.recall)}, F1 {pct(info.score.f1)};{" "}
                {info.score.matched}/{info.score.breaks} labelled breaks matched
                {info.score.double_box_seconds > 0 ? ` · marked ${pct(info.score.double_box_marked / info.score.double_box_seconds)} of double box` : ""}
                {info.score.ad_read_seconds > 0 ? ` · marked ${pct(info.score.ad_read_marked / info.score.ad_read_seconds)} of ad reads` : ""}
              </Text>
            )}
          </Stack>
        )
      }>
      <Button label={`What is ${name}?`} isIconOnly icon={<Text size="sm">ⓘ</Text>} size="sm" variant="ghost" />
    </Popover>
  );
}
