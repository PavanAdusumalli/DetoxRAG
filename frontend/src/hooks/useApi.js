import { useCallback, useEffect, useRef, useState } from "react";

/** Run an async loader whenever `deps` change; ignores stale responses. */
export function useApi(loader, deps) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const seq = useRef(0);

  const run = useCallback(() => {
    const id = ++seq.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    loader()
      .then((data) => id === seq.current && setState({ data, error: null, loading: false }))
      .catch((error) => id === seq.current && setState({ data: null, error, loading: false }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    run();
  }, [run]);

  return { ...state, reload: run };
}

/** Minimal hash router: #/assistant/P015?q=...&hours=12&run=1 ->
 *  { page: "assistant", patientId: "P015", params: { q, hours, run } } */
export function parseHash(hash) {
  const [path, query = ""] = hash.replace(/^#\/?/, "").split("?");
  const [page = "dashboard", patientId = null] = path.split("/");
  return { page: page || "dashboard", patientId: patientId || null, params: Object.fromEntries(new URLSearchParams(query)) };
}

export function useHashRoute() {
  const parse = () => parseHash(window.location.hash);
  const [route, setRoute] = useState(parse);
  useEffect(() => {
    const onChange = () => setRoute(parse());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const navigate = useCallback((page, patientId) => {
    window.location.hash = `/${page}${patientId ? `/${patientId}` : ""}`;
  }, []);
  return [route, navigate];
}
