import { useEffect, useState } from "react";
import { RecordingList } from "./RecordingList";
import { ModelsPage } from "./ModelsPage";
import { RunsPage } from "./RunsPage";
import { Review } from "./Review";

/** Routes by the hash: `#/models`, `#/runs`, `#/<recording id>` reviews a recording, anything else lists them. */
export function App() {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const on = () => setHash(location.hash);
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  if (hash === "#/models" || hash === "#/model") return <ModelsPage />;
  if (hash === "#/runs") return <RunsPage />;
  const m = hash.match(/^#\/(.+)/);
  return m ? <Review key={m[1]} id={decodeURIComponent(m[1])} /> : <RecordingList />;
}
