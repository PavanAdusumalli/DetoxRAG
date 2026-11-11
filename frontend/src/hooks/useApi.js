import { useCallback, useEffect, useRef, useState } from "react";

/** Run an async loader whenever `deps` change; ignores stale responses.
 *
 * `loading` is also true during the render in which `deps` have just changed
 * but the effect that starts the new request has not run yet — otherwise a
 * consumer would briefly see { loading: false, data: <previous result> }
 * for the new deps (e.g. null for a patient that was just selected). */
export function useApi(loader, deps) {
  const key = JSON.stringify(deps);
  const [state, setState] = useState({ data: null, error: null, loading: true, key: null });
  const seq = useRef(0);

  const run = useCallback(() => {
    const id = ++seq.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    loader()
      .then((data) => id === seq.current && setState({ data, error: null, loading: false, key }))
      .catch((error) => id === seq.current && setState({ data: null, error, loading: false, key }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    run();
  }, [run]);

  const stale = state.key !== key;
  return { data: state.data, error: stale ? null : state.error, loading: state.loading || stale, reload: run };
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
