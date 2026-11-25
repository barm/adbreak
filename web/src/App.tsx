import { useEffect, useState } from "react";
import { RecordingList } from "./RecordingList";
import { Review } from "./Review";

/** Routes by the hash: `#/<recording id>` reviews a recording, anything else lists them. */
export function App() {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const on = () => setHash(location.hash);
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  const m = hash.match(/^#\/(.+)/);
  return m ? <Review key={m[1]} id={decodeURIComponent(m[1])} /> : <RecordingList />;
}
