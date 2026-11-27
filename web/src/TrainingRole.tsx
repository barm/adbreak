import { Badge } from "@astryxdesign/core/Badge";
import { Tooltip } from "@astryxdesign/core/Tooltip";

/** A recording's part in the model: held out (scored, never trained on), training, or not trained on (not ready). */
export function TrainingRole({ role }: { role: "held out" | "training" | "not trained on" }) {
  const help = {
    "held out": "In the held-out set: never trained on, and every model is scored on it. Its scores are fair ones.",
    training: "The model trains on this recording, so it isn't scored on it (it would score itself). With no held-out set, scores come from a model trained on the other recordings.",
    "not trained on": "Not ready for training yet. Its model rows are a draft from the installed model, to label against.",
  }[role];
  return (
    <Tooltip content={help}>
      <Badge label={role} variant={role === "held out" ? "purple" : role === "training" ? "blue" : "neutral"} />
    </Tooltip>
  );
}
