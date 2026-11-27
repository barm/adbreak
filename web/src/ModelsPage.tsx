import { Layout, LayoutContent, VStack, HStack } from "@astryxdesign/core/Layout";
import { Heading, Text } from "@astryxdesign/core/Text";
import { Card } from "@astryxdesign/core/Card";
import { Grid } from "@astryxdesign/core/Grid";
import { Table } from "@astryxdesign/core/Table";
import { Button } from "@astryxdesign/core/Button";
import { Link } from "@astryxdesign/core/Link";
import { Divider } from "@astryxdesign/core/Divider";
import { Collapsible } from "@astryxdesign/core/Collapsible";
import { MetadataList, MetadataListItem } from "@astryxdesign/core/MetadataList";
import { Icon } from "@astryxdesign/core/Icon";
import { ArrowDownIcon, ArrowUpIcon } from "@heroicons/react/20/solid";
import { AppHeader } from "./AppHeader";
import { LearningCurve } from "./LearningCurve";
import { HowItWorks } from "./HowItWorks";
import { useModel, kept, modelName, pct, secs, perHour, hours, mb, type Info, type Total } from "./model";

/** One headline number of the installed model, with how it changed from the model it replaced. */
function Metric({ label, value, unseen, change, better, up, help }: { label: string; value: string; unseen?: string; change?: string; better?: boolean; up?: boolean; help: string }) {
  return (
    <Card>
      <VStack gap={1}>
        <Text type="label" color="secondary">{label}</Text>
        <Heading level={2}>{value}</Heading>
        {unseen && <Text type="supporting" color="secondary">{`${unseen} on a network it hasn't seen`}</Text>}
        {change && (
          <HStack gap={1} vAlign="center">
            {better !== undefined && <Icon icon={up ? ArrowUpIcon : ArrowDownIcon} size="xsm" color={better ? "success" : "error"} />}
            <Text type="supporting" color="secondary">{change}</Text>
          </HStack>
        )}
        <Text type="supporting" color="secondary">{help}</Text>
      </VStack>
    </Card>
  );
}

/** The model: the installed one's results, every version with its own, how results grow with data, and how it works. */
export function ModelsPage() {
  const { m, busy, error, run } = useModel();
  const info = m?.info;
  const live = (source: string) => m?.totals.find((t) => t.kind === "commercial break" && t.source === source);
  const now = live("model"), unseen = live("model-new-network");
  const prev = m?.history.find((h) => h.version !== info?.version);
  const was = kept(prev, "model");
  // A change against the previous model's scores from when it was trained (over the recordings labelled then).
  const delta = (cur: number | undefined, old: number | undefined, fmt: (x: number) => string, lowerIsBetter: boolean) => {
    if (cur === undefined || old === undefined || !prev) return {};
    const d = cur - old;
    if (Math.abs(d) < 1e-9) return { change: `same as ${modelName(prev)}` };
    return { change: `${d > 0 ? "+" : "−"}${fmt(Math.abs(d))} vs ${modelName(prev)}`, better: lowerIsBetter ? d < 0 : d > 0, up: d > 0 };
  };
  // Start and end each: "−0.7 / −1.4 s"; better if their sum went down.
  const edgeDelta = (cur: Total, old: Total | undefined) => {
    if (!old || !prev) return {};
    const sign = (d: number) => `${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d).toFixed(1)}`;
    const ds = cur.start_error - old.start_error, de = cur.end_error - old.end_error;
    return { change: `${sign(ds)} / ${sign(de)} s vs ${modelName(prev)}`, better: ds + de < 0, up: ds + de > 0 };
  };
  const versions = m ? [...(info && !m.history.some((h) => h.version === info.version) ? [info] : []), ...m.history] : [];
  const loss = info?.loss;
  const lastOf = (v?: number[]) => (v && v.length ? v[v.length - 1] : undefined);

  return (
    <Layout
      header={<AppHeader section="models" />}
      content={
        <LayoutContent padding={4}>
          <VStack gap={6}>
            {error && <Text color="accent">{error}</Text>}
            {m && (
              <VStack gap={2}>
                <HStack gap={2} vAlign="center" hAlign="between" wrap="wrap">
                  <Heading level={3}>{info ? `Installed: ${modelName(info)}` : "No model installed"}</Heading>
                  <Link href="#/runs">Retrain on the Runs page</Link>
                </HStack>
                {info && (
                  <Text color="secondary">
                    {[info.note, `trained ${new Date(info.exported_at).toLocaleDateString()} on ${info.trained_on.length} recordings (${hours(info.seconds)} labelled)`].filter(Boolean).join(" · ")}
                  </Text>
                )}
              </VStack>
            )}
            {now && (
              <Grid columns={{ minWidth: 220, max: 4, repeat: "fit" }} gap={4}>
                <Metric label="F1, commercial breaks" value={pct(now.f1)} unseen={unseen && pct(unseen.f1)} {...delta(now.f1 * 100, was && was.f1 * 100, (x) => `${x.toFixed(1)} pts`, false)} help={`${now.matched} of ${now.breaks} breaks found`} />
                <Metric label="Edge error, start / end" value={`${secs(now.start_error)} / ${secs(now.end_error)}`} unseen={unseen && `${secs(unseen.start_error)} / ${secs(unseen.end_error)}`} {...edgeDelta(now, was)} help="How far a found break's ends are from the labels" />
                <Metric label="Show cut, per hour" value={perHour(now.wrongly_skipped_per_hour)} unseen={unseen && perHour(unseen.wrongly_skipped_per_hour)} {...delta(now.wrongly_skipped_per_hour, was?.wrongly_skipped_per_hour, perHour, true)} help="Programme a viewer would lose to skipping" />
                <Metric label="Ads left, per hour" value={perHour(now.missed_per_hour)} unseen={unseen && perHour(unseen.missed_per_hour)} {...delta(now.missed_per_hour, was?.missed_per_hour, perHour, true)} help="Break time a viewer would still sit through" />
              </Grid>
            )}
            {m && (
              <Text type="supporting" color="secondary">
                Each recording is scored by a model trained on the others (leave-one-out), over its reviewed time; scores per recording are on the Recordings page.
              </Text>
            )}
            {m && versions.length > 0 && (
              <Card>
                <VStack gap={3}>
                  <Heading level={4}>Versions</Heading>
                  <Text type="supporting" color="secondary">Newest first, each with its scores from when it was trained (over the recordings labelled then). Reinstalling one puts it back for recorders; they pick it up on their own.</Text>
                  <Table
                    data={versions}
                    idKey="version"
                    density="compact"
                    dividers="rows"
                    columns={[
                      { key: "name", header: "Model", renderCell: (h: Info) => <VStack gap={0}><Text type="label">{`${modelName(h)}${h.version === info?.version ? " (installed)" : ""}`}</Text><Text type="code" color="secondary">{h.version.slice(0, 8)}</Text></VStack> },
                      { key: "note", header: "What changed", renderCell: (h: Info) => h.note || "" },
                      { key: "exported_at", header: "Trained", renderCell: (h: Info) => new Date(h.exported_at).toLocaleDateString() },
                      { key: "trained_on", header: "Recordings", align: "end" as const, renderCell: (h: Info) => `${h.trained_on.length}` },
                      { key: "f1", header: "F1", align: "end" as const, renderCell: (h: Info) => (kept(h, "model") ? pct(kept(h, "model")!.f1) : "") },
                      { key: "edges", header: "Start / end error", align: "end" as const, renderCell: (h: Info) => { const t = kept(h, "model"); return t ? `${secs(t.start_error)} / ${secs(t.end_error)}` : ""; } },
                      { key: "cut", header: "Show cut / h", align: "end" as const, renderCell: (h: Info) => (kept(h, "model") ? perHour(kept(h, "model")!.wrongly_skipped_per_hour) : "") },
                      { key: "left", header: "Ads left / h", align: "end" as const, renderCell: (h: Info) => (kept(h, "model") ? perHour(kept(h, "model")!.missed_per_hour) : "") },
                      { key: "nn", header: "New network F1", align: "end" as const, renderCell: (h: Info) => (kept(h, "model-new-network") ? pct(kept(h, "model-new-network")!.f1) : "") },
                      { key: "reinstall", header: "", align: "end" as const, renderCell: (h: Info) => (h.version === info?.version ? null : <Button label="Reinstall" size="sm" variant="secondary" isDisabled={busy} onClick={() => run("install", { version: h.version })} />) },
                    ]}
                  />
                </VStack>
              </Card>
            )}
            {m && (
              <Grid columns={{ minWidth: 420, max: 2, repeat: "fit" }} gap={4}>
                <Card>
                  <VStack gap={3}>
                    <Heading level={4}>All scores</Heading>
                    <Text type="supporting" color="secondary">The installed model's outputs and Comskip, leave-one-out. model-new-network trains only on other networks: the closer test of a channel it hasn't seen.</Text>
                    <Table
                      data={m.totals.map((t, i) => ({ ...t, i }))}
                      idKey="i"
                      density="compact"
                      dividers="rows"
                      columns={[
                        { key: "kind", header: "Kind" },
                        { key: "source", header: "Source" },
                        { key: "f1", header: "F1", align: "end" as const, renderCell: (t: Total) => pct(t.f1) },
                        { key: "matched", header: "Found", align: "end" as const, renderCell: (t: Total) => `${t.matched} of ${t.breaks}` },
                        { key: "edges", header: "Start / end", align: "end" as const, renderCell: (t: Total) => (t.matched ? `${secs(t.start_error)} / ${secs(t.end_error)}` : "") },
                        { key: "cut", header: "Cut / h", align: "end" as const, renderCell: (t: Total) => (t.kind === "commercial break" ? perHour(t.wrongly_skipped_per_hour) : "") },
                        { key: "left", header: "Left / h", align: "end" as const, renderCell: (t: Total) => (t.kind === "commercial break" ? perHour(t.missed_per_hour) : "") },
                      ]}
                    />
                  </VStack>
                </Card>
                <Card>
                  <VStack gap={3}>
                    <HStack hAlign="between" vAlign="center">
                      <Heading level={4}>Learning curve</Heading>
                      <Button label={m.learning_curve ? "Recompute" : "Compute"} size="sm" variant="secondary" isDisabled={busy} tooltip="Train on 1, 2, … recordings and score on the rest (a run, a few minutes)" onClick={() => run("learning-curve")} />
                    </HStack>
                    {m.learning_curve ? (
                      <>
                        <Text type="supporting" color="secondary">{`How the model does with more recordings: trained on k, scored on the ones it left out (${m.learning_curve.fits} models over ${m.learning_curve.recordings} recordings; line: median, band: middle half). Computed ${new Date(m.learning_curve.computed_at).toLocaleDateString()}.`}</Text>
                        <LearningCurve curve={m.learning_curve} />
                      </>
                    ) : (
                      <Text type="supporting" color="secondary">Not computed yet.</Text>
                    )}
                  </VStack>
                </Card>
              </Grid>
            )}
            {m && info && (
              <Card>
                <VStack gap={3}>
                  <Heading level={4}>Details</Heading>
                  <MetadataList>
                    <MetadataListItem label="Version">{info.version}</MetadataListItem>
                    <MetadataListItem label="Training data">{`${info.trained_on.length} recordings, ${hours(info.seconds)} labelled, ${hours(info.break_seconds)} of it commercial breaks; ${Object.entries(info.kind_seconds).map(([k, s]) => `${k.replace("_", " ")} ${Math.round(s / 60)} min`).join(", ")}`}</MetadataListItem>
                    <MetadataListItem label="Trees">{`${info.trees} of up to ${info.max_trees}${loss?.validation ? `; held-out log loss ${lastOf(loss.validation)!.toFixed(3)}, from ${loss.validation[0].toFixed(3)} before the first` : ""}`}</MetadataListItem>
                    <MetadataListItem label="Outputs">{["commercial break", ...m.kinds.map((k) => k.replace("_", " "))].join(", ")}</MetadataListItem>
                    <MetadataListItem label="Size">{`${mb(m.model_bytes)} model + ${mb(m.image_model_bytes)} image model (${m.image_model})`}</MetadataListItem>
                    <MetadataListItem label="Directory">{m.dir}</MetadataListItem>
                  </MetadataList>
                </VStack>
              </Card>
            )}
            {m && (
              <>
                <Divider />
                <Collapsible trigger={<Heading level={4}>How this works</Heading>} defaultIsOpen={false}>
                  <HowItWorks ids={m.recordings.map((r) => r.id)} titles={Object.fromEntries(m.recordings.map((r) => [r.id, r.title]))} />
                </Collapsible>
              </>
            )}
          </VStack>
        </LayoutContent>
      }
    />
  );
}
