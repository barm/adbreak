import { Badge } from "@astryxdesign/core/Badge";
import { Tooltip } from "@astryxdesign/core/Tooltip";

/** Whether the installed model was trained on a recording, as a badge with what that means for its numbers. */
export function TrainingRole({ inTraining }: { inTraining: boolean }) {
  return inTraining ? (
    <Tooltip content="The installed model was trained on this recording. Its scores here come from a separate model trained on the other recordings (held out), so they show how the model does on a show it hasn't seen.">
      <Badge label="training" variant="blue" />
    </Tooltip>
  ) : (
    <Tooltip content="Not in the installed model's training: no reviewed labels yet. Its model rows are a draft from the installed model, to label against.">
      <Badge label="not trained on" variant="neutral" />
    </Tooltip>
  );
}
