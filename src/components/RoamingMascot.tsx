"use client";

import React, { useEffect, useRef, useState } from "react";
import { Pause, Play, X } from "lucide-react";

interface RoamingMascotProps {
  enabled: boolean;
  modelName?: string;
  deviceName: string;
  totalRequiredGB: number;
  usableGB: number;
  decodeTps: number;
  fits: boolean;
}

export function RoamingMascot({
  enabled,
  modelName,
  deviceName,
  totalRequiredGB,
  usableGB,
  decodeTps,
  fits,
}: RoamingMascotProps) {
  const [bubbleOpen, setBubbleOpen] = useState(false);
  const [paused, setPaused] = useState(false);

  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const leftLegRef = useRef<SVGRectElement | null>(null);
  const rightLegRef = useRef<SVGRectElement | null>(null);

  useEffect(() => {
    if (!enabled) return;

    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced || paused) return;

    let rafId = 0;
    let x = 28;
    let dir: 1 | -1 = 1;
    let stridePhase = 0;
    let lastTs = 0;

    // Zero-rerender RAF movement: mutates transform DOM refs directly
    const tick = (now: number) => {
      rafId = window.requestAnimationFrame(tick);
      if (document.hidden) {
        lastTs = 0;
        return;
      }

      const dt =
        lastTs === 0 ? 0.016 : Math.max(0, Math.min(0.05, (now - lastTs) / 1000));
      lastTs = now;

      const maxX = Math.max(48, window.innerWidth - 96);
      x += dir * 38 * dt;
      if (x >= maxX) {
        x = maxX;
        dir = -1;
      } else if (x <= 20) {
        x = 20;
        dir = 1;
      }

      stridePhase += dt * 7.5;
      const bobY = Math.abs(Math.sin(stridePhase)) * -3.5;
      const leftLift = Math.max(0, Math.sin(stridePhase)) * -3;
      const rightLift = Math.max(0, -Math.sin(stridePhase)) * -3;

      if (wrapperRef.current) {
        wrapperRef.current.style.transform = `translate3d(${x.toFixed(1)}px, 0, 0)`;
      }
      if (svgRef.current) {
        svgRef.current.style.transform = `translate3d(0, ${bobY.toFixed(1)}px, 0) scaleX(${dir})`;
      }
      if (leftLegRef.current) {
        leftLegRef.current.setAttribute("y", (34 + leftLift).toFixed(1));
      }
      if (rightLegRef.current) {
        rightLegRef.current.setAttribute("y", (34 + rightLift).toFixed(1));
      }
    };

    rafId = window.requestAnimationFrame(tick);
    return () => {
      window.cancelAnimationFrame(rafId);
    };
  }, [enabled, paused]);

  if (!enabled) return null;

  return (
    <div
      ref={wrapperRef}
      className="pointer-events-none fixed bottom-3 left-0 z-40"
      style={{ transform: "translate3d(28px, 0, 0)", willChange: "transform" }}
    >
      <div className="pointer-events-auto relative flex flex-col items-start">
        {bubbleOpen && (
          <div className="surface mb-2 w-64 rounded-2xl p-3 text-xs shadow-lg">
            <div className="flex items-start justify-between gap-2">
              <span className="font-mono text-[10px] font-bold uppercase text-[var(--soft-blue)]">
                M-01 Tensor Scout
              </span>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => setPaused((p) => !p)}
                  aria-label={paused ? "Resume mascot patrol" : "Pause mascot patrol"}
                  className="muted rounded p-0.5 hover:text-[var(--soft-text)]"
                >
                  {paused ? <Play className="h-3 w-3" /> : <Pause className="h-3 w-3" />}
                </button>
                <button
                  type="button"
                  onClick={() => setBubbleOpen(false)}
                  aria-label="Close mascot telemetry bubble"
                  className="muted rounded p-0.5 hover:text-[var(--soft-text)]"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            </div>

            <p className="mt-1 truncate font-bold">
              {modelName ?? "No model selected"}
            </p>
            <p className="muted truncate font-mono text-[10px]">{deviceName}</p>
            <div className="well mt-2 flex items-center justify-between rounded-xl px-2.5 py-1.5 font-mono text-[10px] tabular">
              <span className={fits ? "stat--up font-bold" : "stat--warn font-bold"}>
                {fits ? "FITS" : "OOM"}
              </span>
              <span>
                {totalRequiredGB.toFixed(1)}/{usableGB.toFixed(1)} GB
              </span>
              {decodeTps > 0 && <span>{decodeTps} t/s</span>}
            </div>
          </div>
        )}

        <button
          type="button"
          onClick={() => setBubbleOpen((o) => !o)}
          aria-label="Toggle M-01 companion telemetry bubble"
          className="surface flex h-12 w-12 items-center justify-center rounded-2xl p-1"
        >
          <svg
            ref={svgRef}
            width="40"
            height="42"
            viewBox="0 0 40 42"
            fill="none"
            aria-hidden="true"
          >
            {/* Antenna */}
            <line
              x1="20"
              y1="3"
              x2="20"
              y2="9"
              stroke="var(--soft-blue)"
              strokeWidth="2.2"
              strokeLinecap="round"
            />
            <circle
              cx="20"
              cy="3"
              r="2.4"
              fill={fits ? "var(--soft-green)" : "var(--soft-orange)"}
            />
            {/* Soft Ceramic Body */}
            <rect
              x="7"
              y="9"
              width="26"
              height="24"
              rx="8"
              fill="var(--soft-bg)"
              stroke="var(--soft-blue)"
              strokeWidth="2.2"
            />
            {/* Visor */}
            <rect x="11" y="14" width="18" height="9" rx="4" fill="#0f172a" />
            {/* Expressive Eyes */}
            <circle
              cx="16"
              cy="18.5"
              r="2"
              fill={fits ? "#22c55e" : "#fb923c"}
            />
            <circle
              cx="24"
              cy="18.5"
              r="2"
              fill={fits ? "#22c55e" : "#fb923c"}
            />
            {/* Chest Status Core */}
            <rect
              x="14"
              y="26"
              width="12"
              height="3.2"
              rx="1.6"
              fill={fits ? "#22c55e" : "#ea580c"}
            />
            {/* Articulated Legs */}
            <rect
              ref={leftLegRef}
              x="12"
              y="34"
              width="5"
              height="5"
              rx="2"
              fill="var(--soft-blue)"
            />
            <rect
              ref={rightLegRef}
              x="23"
              y="34"
              width="5"
              height="5"
              rx="2"
              fill="var(--soft-blue)"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}
