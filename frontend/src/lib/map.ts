import * as maplibregl from "maplibre-gl";
import type { StyleSpecification } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { useEffect, useState } from "react";

// MapLibre v6 loads its tile worker from a separate file; let Vite bundle it and tell MapLibre where it is.
maplibregl.setWorkerUrl(workerUrl);

/** Keyless vector basemaps from OpenFreeMap (OpenStreetMap data), light and dark to match the app theme. */
export function baseStyle(dark: boolean): string | StyleSpecification {
  return `https://tiles.openfreemap.org/styles/${dark ? "dark" : "positron"}`;
}

/** Tracks the app's dark-mode class so maps can swap tiles. */
export function useDarkClass() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  useEffect(() => {
    const obs = new MutationObserver(() => setDark(document.documentElement.classList.contains("dark")));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return dark;
}

export const KIND_COLOR: Record<string, string> = {
  unesco: "#b2472d",
  repository: "#2e4374",
  temple: "#9a6512",
  stepwell: "#2f7a4b",
  monument: "#6b4f9a",
  museum: "#3b7a8a",
  fort: "#7a5a3b",
  other: "#7a7064",
};

export const KIND_LABEL: Record<string, string> = {
  unesco: "UNESCO World Heritage site",
  repository: "Manuscript library / repository",
  temple: "Temple",
  stepwell: "Stepwell",
  monument: "Monument",
  museum: "Museum",
  fort: "Fort",
  other: "Heritage site",
};

/**
 * A map marker.  MapLibre positions the element it is given with an inline `transform`, so the hover
 * effect must NOT touch that element (scaling it made the dot jump away from its place).  The returned
 * wrapper is positioned by MapLibre; only the inner dot is scaled.
 */
export function markerEl(kind: string, label?: string, active = false) {
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;cursor:pointer";
  const dot = document.createElement("button");
  dot.type = "button";
  dot.setAttribute("aria-label", label ?? kind);
  const color = KIND_COLOR[kind] ?? KIND_COLOR.other;
  dot.style.cssText = `width:${active ? 30 : 22}px;height:${active ? 30 : 22}px;border-radius:999px;background:${color};border:3px solid white;box-shadow:0 2px 8px rgb(0 0 0 / .35);cursor:pointer;display:flex;align-items:center;justify-content:center;color:white;font:600 11px Inter Variable, sans-serif;transition:transform .15s;padding:0`;
  if (label) dot.textContent = label;
  wrap.append(dot);
  wrap.addEventListener("mouseenter", () => (dot.style.transform = "scale(1.2)"));
  wrap.addEventListener("mouseleave", () => (dot.style.transform = ""));
  return wrap;
}

interface SiteInfo {
  name: string;
  kind: string;
  description?: string;
  city?: string;
  state?: string;
  era?: string;
  visit_minutes?: number;
  manuscript_count?: number;
}

/** The hover card for a heritage site (built with textContent, never innerHTML). */
export function siteCard(s: SiteInfo): HTMLElement {
  const card = document.createElement("div");
  card.style.cssText = "max-width:260px;font:13px/1.45 Inter Variable, sans-serif";
  const add = (tag: string, text: string, css: string) => {
    const e = document.createElement(tag);
    e.textContent = text;
    e.style.cssText = css;
    card.append(e);
  };
  add("div", s.name, "font-weight:600;font-size:14px;margin-bottom:2px");
  add("div", KIND_LABEL[s.kind] ?? KIND_LABEL.other, `color:${KIND_COLOR[s.kind] ?? KIND_COLOR.other};font-weight:500;font-size:12px;margin-bottom:4px`);
  const where = [s.city, s.state].filter(Boolean).join(", ");
  if (where) add("div", where, "opacity:.75;font-size:12px");
  const facts = [s.era, s.visit_minutes ? `about ${s.visit_minutes} min visit` : "", s.manuscript_count ? `${s.manuscript_count} manuscript${s.manuscript_count === 1 ? "" : "s"} here` : ""].filter(Boolean);
  if (facts.length) add("div", facts.join(" · "), "opacity:.75;font-size:12px;margin-top:2px");
  if (s.description) add("div", s.description.length > 170 ? s.description.slice(0, 167) + "…" : s.description, "margin-top:6px;font-size:12px");
  add("div", "Click for details and to add to a trail", "margin-top:6px;font-size:11px;opacity:.6");
  return card;
}

export function fitTo(map: maplibregl.Map, points: [number, number][], padding = 70) {
  if (!points.length) return;
  const b = new maplibregl.LngLatBounds(points[0], points[0]);
  points.forEach((p) => b.extend(p));
  map.fitBounds(b, { padding, maxZoom: 13, duration: 800 });
}
