import { describe, expect, it } from "vitest";
import { fmtHours, fmtPct, fmtSigned, fmtTime, isGuidelineId, niceTicks, parseTs, splitCitations, statusMeta } from "./format.js";

describe("timestamps", () => {
  it("parses facility-local ISO timestamps without shifting time zone", () => {
    const d = parseTs("2026-03-02T14:37:00");
    expect([d.getFullYear(), d.getMonth(), d.getDate(), d.getHours(), d.getMinutes()]).toEqual([2026, 2, 2, 14, 37]);
    expect(fmtTime("2026-03-02T04:05:00")).toBe("Mar 2, 04:05");
    expect(parseTs("garbage")).toBeNull();
    expect(fmtTime(null)).toBe("—");
  });
});

describe("numbers", () => {
  it("formats hours, signed values and percentages", () => {
    expect(fmtHours(4)).toBe("4 h");
    expect(fmtHours(4.38)).toBe("4.38 h");
    expect(fmtHours(20.88)).toBe("20.9 h");
    expect(fmtSigned(11)).toBe("+11");
    expect(fmtSigned(-3)).toBe("-3");
    expect(fmtSigned(null)).toBe("—");
    expect(fmtPct(0.9333)).toBe("93.3%");
    expect(fmtPct(null)).toBe("—");
  });
});

describe("niceTicks", () => {
  it("produces round, evenly spaced ticks that cover the range", () => {
    expect(niceTicks(0, 30, 4)).toEqual([0, 10, 20, 30]);
    expect(niceTicks(97.4, 99.1, 4)).toEqual([97, 97.5, 98, 98.5, 99, 99.5]);
    const t = niceTicks(64, 105, 4);
    expect(t[0]).toBeLessThanOrEqual(64);
    expect(t.at(-1)).toBeGreaterThanOrEqual(105);
    const steps = t.slice(1).map((v, i) => v - t[i]);
    expect(new Set(steps).size).toBe(1);
  });
});

describe("splitCitations", () => {
  it("splits single and multi-ID brackets", () => {
    const parts = splitCitations("Score rose [CALC-SCORE]. Doses [CALC-MEDS, GAP-02] noted.");
    expect(parts.filter((p) => p.type === "cite").map((p) => p.value)).toEqual(["CALC-SCORE", "CALC-MEDS", "GAP-02"]);
    expect(parts[0]).toEqual({ type: "text", value: "Score rose " });
  });
  it("leaves non-citation brackets as text", () => {
    expect(splitCitations("[unsupported citation removed]")).toEqual([{ type: "text", value: "[unsupported citation removed]" }]);
  });
  it("distinguishes guideline IDs from patient evidence IDs", () => {
    expect(isGuidelineId("BJA-2023-0071")).toBe(true);
    expect(isGuidelineId("ASM-00148")).toBe(false);
    expect(isGuidelineId("CALC-SCORE")).toBe(false);
  });
});

describe("parseHash", () => {
  it("parses page, patient and query parameters", async () => {
    const { parseHash } = await import("../hooks/useApi.js");
    expect(parseHash("#/assistant/P015?q=hello%20world&run=1")).toEqual({
      page: "assistant", patientId: "P015", params: { q: "hello world", run: "1" },
    });
    expect(parseHash("")).toEqual({ page: "dashboard", patientId: null, params: {} });
  });
});

describe("statusMeta", () => {
  it("pairs every status with an icon and label", () => {
    for (const s of ["Administered", "Held", "Refused", "Not administered", "Not documented"]) {
      const m = statusMeta(s);
      expect(m.icon).toBeTruthy();
      expect(m.label).toBe(s);
    }
    expect(statusMeta("Not documented").tone).toBe("critical");
    expect(statusMeta("Administered").tone).toBe("good");
  });
});
