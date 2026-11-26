"use client";

import React, { useId, useMemo, useSyncExternalStore } from "react";
import { Download, Shuffle } from "lucide-react";

const STORAGE_KEY = "monolith-logo-seed";
const DEFAULT_SEED = 4096;

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface EmblemGeometry {
  variant: number;
  rings: number;
  spokes: number;
  rotationDeg: number;
  innerPolygonSides: number;
  innerScale: number;
  matrixDots: { x: number; y: number; active: boolean }[];
}

function buildGeometry(seed: number): EmblemGeometry {
  const rng = mulberry32(seed);
  const variant = seed === DEFAULT_SEED ? 0 : Math.floor(rng() * 4);
  const rings = 2 + Math.floor(rng() * 2);
  const spokes = [4, 6, 8, 12][Math.floor(rng() * 4)];
  const rotationDeg = seed === DEFAULT_SEED ? 0 : Math.floor(rng() * 6) * 15;
  const innerPolygonSides = [4, 6, 8][Math.floor(rng() * 3)];
  const innerScale = 0.42 + rng() * 0.2;

  const matrixDots: { x: number; y: number; active: boolean }[] = [];
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const center = row === 1 && col === 1;
      matrixDots.push({
        x: 20 + col * 12,
        y: 20 + row * 12,
        active: center || rng() > 0.42,
      });
    }
  }

  return {
    variant,
    rings,
    spokes,
    rotationDeg,
    innerPolygonSides,
    innerScale,
    matrixDots,
  };
}

function polygonPoints(
  cx: number,
  cy: number,
  radius: number,
  sides: number,
  offsetDeg = -90
): string {
  const pts: string[] = [];
  for (let i = 0; i < sides; i++) {
    const angle = ((offsetDeg + (360 / sides) * i) * Math.PI) / 180;
    pts.push(
      `${(cx + Math.cos(angle) * radius).toFixed(2)},${(cy + Math.sin(angle) * radius).toFixed(2)}`
    );
  }
  return pts.join(" ");
}

const logoListeners = new Set<() => void>();

function subscribeLogoSeed(callback: () => void): () => void {
  logoListeners.add(callback);
  return () => {
    logoListeners.delete(callback);
  };
}

function getLogoSeedSnapshot(): number {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed = parseInt(saved, 10);
      if (Number.isFinite(parsed) && parsed > 0) return parsed;
    }
  } catch {
    // Ignore storage access issues
  }
  return DEFAULT_SEED;
}

function getLogoSeedServerSnapshot(): number {
  return DEFAULT_SEED;
}

export function GeneratedLogo() {
  const seed = useSyncExternalStore(
    subscribeLogoSeed,
    getLogoSeedSnapshot,
    getLogoSeedServerSnapshot
  );
  const gradId = useId().replace(/:/g, "");

  const geom = useMemo(() => buildGeometry(seed), [seed]);

  const handleRandomize = () => {
    const nextSeed = Math.floor(1000 + Math.random() * 899999);
    try {
      window.localStorage.setItem(STORAGE_KEY, String(nextSeed));
    } catch {
      // Ignore storage failures
    }
    logoListeners.forEach((cb) => cb());
  };

  const handleDownloadSvg = () => {
    const el = document.getElementById(`monolith-emblem-${gradId}`);
    if (!el) return;
    const serialized = new XMLSerializer().serializeToString(el);
    const blob = new Blob([serialized], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `monolith-logo-${seed}.svg`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex items-center gap-3">
      <div className="surface flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl p-1">
        <svg
          id={`monolith-emblem-${gradId}`}
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 64 64"
          className="h-10 w-10"
          role="img"
          aria-label={`MONOLITH geometric emblem seed ${seed}`}
        >
          <defs>
            <linearGradient id={`prism-${gradId}`} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#2563EB" />
              <stop offset="55%" stopColor="#0891B2" />
              <stop offset="100%" stopColor="#22C55E" />
            </linearGradient>
            <linearGradient id={`warm-${gradId}`} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#F59E0B" />
              <stop offset="100%" stopColor="#EA580C" />
            </linearGradient>
            <linearGradient id={`plate-${gradId}`} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#1F2937" />
              <stop offset="100%" stopColor="#111827" />
            </linearGradient>
          </defs>

          <rect
            x="3"
            y="3"
            width="58"
            height="58"
            rx="15"
            fill={`url(#plate-${gradId})`}
            stroke="rgba(255,255,255,0.22)"
            strokeWidth="1.25"
          />

          <g transform={`rotate(${geom.rotationDeg} 32 32)`}>
            {geom.variant === 0 && (
              <>
                <polygon
                  points={polygonPoints(32, 32, 20, 6)}
                  fill="#0F172A"
                  stroke={`url(#prism-${gradId})`}
                  strokeWidth="2.2"
                />
                <polygon
                  points={polygonPoints(32, 32, 12.5, 6)}
                  fill="none"
                  stroke={`url(#warm-${gradId})`}
                  strokeWidth="1.4"
                />
                <line x1="32" y1="12" x2="32" y2="52" stroke="#60A5FA" strokeWidth="1.1" opacity="0.7" />
                <line x1="14.7" y1="22" x2="49.3" y2="42" stroke="#22C55E" strokeWidth="1" opacity="0.5" />
                <line x1="49.3" y1="22" x2="14.7" y2="42" stroke="#EA580C" strokeWidth="1" opacity="0.5" />
                <circle cx="32" cy="32" r="4.6" fill="#FFFFFF" />
              </>
            )}

            {geom.variant === 1 && (
              <>
                {Array.from({ length: geom.rings }).map((_, idx) => (
                  <circle
                    key={idx}
                    cx="32"
                    cy="32"
                    r={20 - idx * 5.5}
                    fill="none"
                    stroke={idx === 0 ? `url(#prism-${gradId})` : `url(#warm-${gradId})`}
                    strokeWidth={idx === 0 ? 2 : 1.3}
                    strokeDasharray={idx === 1 ? "4 3" : undefined}
                  />
                ))}
                {Array.from({ length: geom.spokes }).map((_, idx) => {
                  const angle = ((360 / geom.spokes) * idx * Math.PI) / 180;
                  return (
                    <line
                      key={idx}
                      x1={(32 + Math.cos(angle) * 7).toFixed(2)}
                      y1={(32 + Math.sin(angle) * 7).toFixed(2)}
                      x2={(32 + Math.cos(angle) * 20).toFixed(2)}
                      y2={(32 + Math.sin(angle) * 20).toFixed(2)}
                      stroke="#60A5FA"
                      strokeWidth="1.2"
                      opacity="0.7"
                    />
                  );
                })}
                <polygon
                  points={polygonPoints(32, 32, 6.2, geom.innerPolygonSides)}
                  fill="#FFFFFF"
                />
              </>
            )}

            {geom.variant === 2 && (
              <>
                <polygon
                  points={polygonPoints(32, 32, 21, geom.innerPolygonSides, 45)}
                  fill="#111827"
                  stroke={`url(#prism-${gradId})`}
                  strokeWidth="2.1"
                />
                <polygon
                  points={polygonPoints(32, 32, 21 * geom.innerScale, geom.innerPolygonSides, 0)}
                  fill="#1E293B"
                  stroke={`url(#warm-${gradId})`}
                  strokeWidth="1.6"
                />
                <rect x="29" y="29" width="6" height="6" rx="1.5" fill="#FFFFFF" />
              </>
            )}

            {geom.variant === 3 && (
              <>
                <rect
                  x="13"
                  y="13"
                  width="38"
                  height="38"
                  rx="8"
                  fill="#111827"
                  stroke={`url(#prism-${gradId})`}
                  strokeWidth="1.9"
                />
                {geom.matrixDots.map((dot, idx) => (
                  <rect
                    key={idx}
                    x={dot.x - 3.3}
                    y={dot.y - 3.3}
                    width="6.6"
                    height="6.6"
                    rx="2"
                    fill={dot.active ? (idx % 2 === 0 ? "#38BDF8" : "#F59E0B") : "#334155"}
                    opacity={dot.active ? 0.98 : 0.7}
                  />
                ))}
              </>
            )}
          </g>
        </svg>
      </div>

      <div className="flex flex-col">
        <div className="flex items-center gap-2">
          <span className="text-base font-bold tracking-tight text-[var(--soft-text)]">
            MONOLITH
          </span>
          <span className="well rounded-lg px-2 py-0.5 font-mono text-[10px] font-bold tracking-wider text-[var(--soft-blue)]">
            LOCALAI
          </span>
        </div>
        <div className="mt-1 flex items-center gap-1.5">
          <span className="muted text-[11px] font-medium">
            Soft UI Inference Studio
          </span>
          <button
            type="button"
            onClick={handleRandomize}
            title={`Generate random geometric logo (seed #${seed})`}
            aria-label="Generate random geometric logo"
            className="button button--neutral inline-flex items-center gap-1 rounded-lg px-2 py-0.5 font-mono text-[10px]"
          >
            <Shuffle className="h-2.5 w-2.5 text-[var(--soft-blue)]" />
            <span>Logo</span>
          </button>
          <button
            type="button"
            onClick={handleDownloadSvg}
            title="Download current SVG emblem"
            aria-label="Download generated SVG logo"
            className="button button--neutral inline-flex items-center rounded-lg p-1"
          >
            <Download className="h-2.5 w-2.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
