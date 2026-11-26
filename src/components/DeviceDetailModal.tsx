"use client";

import React, { useEffect, useMemo } from "react";
import {
  BookOpen,
  CheckCircle2,
  Cpu,
  Gauge,
  HardDrive,
  Layers,
  TriangleAlert,
  X,
  Zap,
} from "lucide-react";
import type {
  HardwareDevice,
  HFModelDetail,
  HFModelSummary,
  InferenceConfig,
  QuantizationLevel,
} from "@/types/inference";
import {
  calculateInference,
  inferArchitectureFallback,
  QUANTIZATION_SPECS,
} from "@/lib/inference-math";

interface DeviceDetailModalProps {
  open: boolean;
  device: HardwareDevice;
  model: HFModelSummary | HFModelDetail;
  config: InferenceConfig;
  onChangeConfig: (updater: (prev: InferenceConfig) => InferenceConfig) => void;
  onOpenHelp: () => void;
  onClose: () => void;
}

export function DeviceDetailModal({
  open,
  device,
  model,
  config,
  onChangeConfig,
  onOpenHelp,
  onClose,
}: DeviceDetailModalProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const arch = useMemo(() => inferArchitectureFallback(model), [model]);
  const estimate = useMemo(
    () => calculateInference(model, device, config),
    [model, device, config]
  );

  if (!open) return null;

  const maxModelContext = Math.max(4096, Math.min(262144, arch.maxContextLength || 32768));

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-3 backdrop-blur-md"
      role="dialog"
      aria-modal="true"
      aria-label={`Hardware device specifications for ${device.name}`}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="surface soft-scroll max-h-[92vh] w-[95vw] max-w-6xl overflow-y-auto p-6">
        {/* Header */}
        <div className="flex flex-wrap items-start justify-between gap-4 border-b border-[var(--soft-border)] pb-4">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-lg bg-[var(--soft-blue-fill)] px-2.5 py-0.5 font-mono text-[10px] font-bold uppercase text-white">
                {device.vendorLabel}
              </span>
              <span className="well rounded-lg px-2.5 py-0.5 font-mono text-[11px] font-semibold text-[var(--soft-text)]">
                {device.architecture} · {device.processNode}
              </span>
              <span className="muted font-mono text-[11px]">
                Released {device.releaseYear} · {device.formFactor}
              </span>
            </div>
            <h2 className="mt-1.5 text-xl font-bold text-[var(--soft-text)]">
              {device.name} — Complete AI Hardware Specifications
            </h2>
            <p className="muted mt-1 text-xs">{device.notes}</p>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={() => {
                onClose();
                onOpenHelp();
              }}
              className="button inline-flex items-center gap-1.5 px-3.5 py-2 text-xs"
            >
              <BookOpen className="h-3.5 w-3.5" />
              <span>Explain Parameters</span>
            </button>

            <button
              type="button"
              onClick={onClose}
              className="button button--primary inline-flex items-center gap-1 px-4 py-2 text-xs"
              aria-label="Close device parameters modal"
            >
              <X className="h-4 w-4" />
              <span>Close</span>
            </button>
          </div>
        </div>

        {/* 3 Hardware Parameter Pillars (1-level .well inside .surface) */}
        <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-3">
          {/* GPU / Compute */}
          <section className="well p-4">
            <div className="mb-3 flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-[var(--soft-blue-fill)] text-white">
                <Zap className="h-4 w-4" />
              </div>
              <div>
                <div className="muted font-mono text-[10px] font-semibold uppercase tracking-wider">
                  GPU & Matrix Compute
                </div>
                <h3 className="text-sm font-bold text-[var(--soft-text)]">
                  Execution Engines
                </h3>
              </div>
            </div>

            <dl className="space-y-2 font-mono text-xs">
              <div className="flex justify-between gap-2">
                <dt className="muted">Compute Cores</dt>
                <dd className="text-right font-semibold text-[var(--soft-text)]">
                  {device.computeCores}
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="muted">FP16 / BF16</dt>
                <dd className="tabular font-semibold text-[var(--soft-text)]">
                  {device.fp16Tflops} TFLOPS
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="muted">FP8 Tensor</dt>
                <dd className="tabular font-semibold text-[var(--soft-text)]">
                  {device.fp8Tflops} TFLOPS
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="muted">INT4 / Q4 Peak</dt>
                <dd className="tabular font-semibold text-[var(--soft-text)]">
                  {device.int4Tops} TOPS
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="muted">Power Envelope (TDP)</dt>
                <dd className="tabular font-semibold text-[var(--soft-text)]">
                  {device.tdpWatts} W
                </dd>
              </div>
            </dl>
          </section>

          {/* VRAM / Unified Memory */}
          <section className="well p-4">
            <div className="mb-3 flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-[#15803D] text-white">
                <HardDrive className="h-4 w-4" />
              </div>
              <div>
                <div className="muted font-mono text-[10px] font-semibold uppercase tracking-wider">
                  VRAM & Memory Fabric
                </div>
                <h3 className="text-sm font-bold text-[var(--soft-text)]">
                  Capacity & Bandwidth
                </h3>
              </div>
            </div>

            <dl className="space-y-2 font-mono text-xs">
              <div className="flex justify-between gap-2">
                <dt className="muted">Total Capacity</dt>
                <dd className="tabular font-semibold text-[var(--soft-text)]">
                  {device.vramGB} GB ({device.unifiedMemory ? "Unified" : "Dedicated"})
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="muted">Effective AI Pool</dt>
                <dd className="tabular stat--up font-bold">
                  {estimate.effectiveUsableMemoryGB.toFixed(1)} GB
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="muted">Memory Type</dt>
                <dd className="text-right font-semibold text-[var(--soft-text)]">
                  {device.memoryType}
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="muted">Bus Width</dt>
                <dd className="tabular font-semibold text-[var(--soft-text)]">
                  {device.memoryBusBits}-bit
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="muted">Memory Bandwidth</dt>
                <dd className="tabular font-bold text-[var(--soft-blue)]">
                  {device.bandwidthGBs.toLocaleString()} GB/s
                </dd>
              </div>
            </dl>
          </section>

          {/* CPU, Host RAM & Interconnect */}
          <section className="well p-4">
            <div className="mb-3 flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-[#C2410C] text-white">
                <Cpu className="h-4 w-4" />
              </div>
              <div>
                <div className="muted font-mono text-[10px] font-semibold uppercase tracking-wider">
                  CPU, Host RAM & Bus
                </div>
                <h3 className="text-sm font-bold text-[var(--soft-text)]">
                  System Pairing
                </h3>
              </div>
            </div>

            <dl className="space-y-2 font-mono text-xs">
              <div className="flex justify-between gap-2">
                <dt className="muted">CPU / Host RAM</dt>
                <dd className="text-right font-semibold text-[var(--soft-text)]">
                  {device.cpuRamRecommendation}
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="muted">Interconnect</dt>
                <dd className="text-right font-semibold text-[var(--soft-text)]">
                  {device.interconnect}
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="muted">Multi-GPU Support</dt>
                <dd className="font-semibold text-[var(--soft-text)]">
                  {device.supportsMultiGpu ? `Up to ${device.maxGpuCount}x` : "Single Node"}
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="muted">Usable Allocation</dt>
                <dd className="tabular font-semibold text-[var(--soft-text)]">
                  {Math.round(device.defaultUsableRatio * 100)}% default /{" "}
                  {Math.round(device.maxUsableRatio * 100)}% max
                </dd>
              </div>
            </dl>

            <div className="mt-3 flex flex-wrap gap-1.5 border-t border-[var(--soft-border)] pt-2.5">
              {device.supportedBackends.map((b) => (
                <span
                  key={b}
                  className="rounded-lg border border-[var(--soft-border)] px-2 py-0.5 font-mono text-[10px] font-semibold text-[var(--soft-text)]"
                >
                  {b}
                </span>
              ))}
            </div>
          </section>
        </div>

        {/* Live Context Window Fit Checker for Selected Model on This Device */}
        <section className="mt-6 border-t border-[var(--soft-border)] pt-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Gauge className="h-4 w-4 text-[var(--soft-blue)]" />
              <div>
                <div className="font-mono text-[10px] font-bold uppercase tracking-wider text-[var(--soft-blue)]">
                  Live Context Window Fit Checker
                </div>
                <h3 className="text-sm font-bold text-[var(--soft-text)]">
                  Testing <span className="font-mono">{model.id}</span> on {device.name}
                </h3>
              </div>
            </div>

            <div
              className={`inline-flex items-center gap-1.5 rounded-xl px-3.5 py-1.5 font-mono text-xs font-bold ${
                estimate.supported ? "badge--ok" : "badge--warn"
              }`}
            >
              {estimate.supported ? (
                <CheckCircle2 className="h-4 w-4 text-[var(--soft-green)]" />
              ) : (
                <TriangleAlert className="h-4 w-4 text-[var(--soft-orange)]" />
              )}
              <span>
                {estimate.supported
                  ? `FITS · +${estimate.headroomGB.toFixed(1)} GB FREE`
                  : estimate.unsupportedReason ?? "DOES NOT FIT"}
              </span>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-5 md:grid-cols-12">
            <div className="space-y-3.5 md:col-span-7">
              <div>
                <div className="mb-1.5 flex items-center justify-between text-xs">
                  <span className="muted font-semibold">Context Window Size</span>
                  <span className="tabular font-mono font-bold text-[var(--soft-text)]">
                    {config.contextLength.toLocaleString()} / {maxModelContext.toLocaleString()} tokens
                  </span>
                </div>
                <input
                  type="range"
                  min={1024}
                  max={maxModelContext}
                  step={1024}
                  value={Math.min(config.contextLength, maxModelContext)}
                  onChange={(e) =>
                    onChangeConfig((prev) => ({
                      ...prev,
                      contextLength: parseInt(e.target.value, 10),
                    }))
                  }
                  className="soft-range w-full"
                  aria-label="Adjust context window inside device specs"
                />
                <div className="mt-2.5 flex flex-wrap gap-2">
                  {[2048, 4096, 8192, 16384, 32768, 65536, 131072]
                    .filter((p) => p <= maxModelContext)
                    .map((preset) => (
                      <button
                        key={preset}
                        type="button"
                        onClick={() =>
                          onChangeConfig((prev) => ({ ...prev, contextLength: preset }))
                        }
                        className={`button rounded-lg px-2.5 py-1 font-mono text-[10px] ${
                          config.contextLength === preset
                            ? "button--primary"
                            : "button--neutral"
                        }`}
                      >
                        {preset / 1024}K
                      </button>
                    ))}
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <span className="muted mr-1 font-mono text-[10px] font-bold">QUANT:</span>
                {QUANTIZATION_SPECS.slice(0, 6).map((q) => (
                  <button
                    key={q.id}
                    type="button"
                    onClick={() =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        quantization: q.id as QuantizationLevel,
                      }))
                    }
                    className={`button rounded-lg px-2.5 py-1 font-mono text-[10px] ${
                      config.quantization === q.id ? "button--primary" : "button--neutral"
                    }`}
                  >
                    {q.id}
                  </button>
                ))}
              </div>
            </div>

            <div className="well flex flex-col justify-between p-4 md:col-span-5">
              <div className="space-y-1.5 font-mono text-xs">
                <div className="flex justify-between">
                  <span className="muted">Weights ({config.quantization})</span>
                  <span className="tabular font-semibold text-[var(--soft-text)]">
                    {estimate.weightsGB.toFixed(2)} GB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="muted">KV Cache</span>
                  <span className="tabular stat--warn font-semibold">
                    {estimate.kvCacheGB.toFixed(2)} GB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="muted">Total / Pool</span>
                  <span className="tabular font-bold text-[var(--soft-text)]">
                    {estimate.totalRequiredGB.toFixed(2)} /{" "}
                    {estimate.effectiveUsableMemoryGB.toFixed(1)} GB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="muted">Max Context on Device</span>
                  <span className="tabular stat--up font-bold">
                    {estimate.maxContextOnDevice.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="muted">Estimated Speed</span>
                  <span className="tabular font-bold text-[var(--soft-blue)]">
                    {estimate.estimatedDecodeTps} tok/s
                  </span>
                </div>
              </div>

              {!estimate.supported && estimate.maxContextOnDevice >= 1024 && (
                <button
                  type="button"
                  onClick={() =>
                    onChangeConfig((prev) => ({
                      ...prev,
                      contextLength: estimate.maxContextOnDevice,
                    }))
                  }
                  className="button button--warn mt-3 inline-flex w-full items-center justify-center gap-1.5 py-2 font-mono text-[11px]"
                >
                  <Layers className="h-3.5 w-3.5" />
                  <span>Snap Context to Max Fit ({estimate.maxContextOnDevice.toLocaleString()})</span>
                </button>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
