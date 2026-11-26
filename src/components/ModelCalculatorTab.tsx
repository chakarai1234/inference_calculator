"use client";

import React, { useMemo, useState } from "react";
import {
  Box,
  Check,
  CheckCircle2,
  Copy,
  Cpu,
  ExternalLink,
  Gauge,
  Info,
  Layers,
  Loader2,
  ServerCrash,
  SlidersHorizontal,
} from "lucide-react";
import type {
  HardwareDevice,
  HFModelDetail,
  HFModelSummary,
  InferenceConfig,
  KVCachePrecision,
  QuantizationLevel,
} from "@/types/inference";
import { getHardwareGroupedByVendor, HARDWARE_CATALOG } from "@/lib/hardware-catalog";
import {
  calculateInference,
  formatBytes,
  formatCompactNumber,
  formatParams,
  generateLocalAIYaml,
  inferArchitectureFallback,
  QUANTIZATION_SPECS,
} from "@/lib/inference-math";
import { ModelFuzzyPicker } from "@/components/ModelFuzzyPicker";

interface ModelCalculatorTabProps {
  models: HFModelSummary[];
  selectedModel: HFModelSummary;
  modelDetail: HFModelDetail | null;
  isLoadingDetail: boolean;
  selectedDevice: HardwareDevice;
  config: InferenceConfig;
  onSelectModel: (model: HFModelSummary) => void;
  onSelectDeviceId: (id: string) => void;
  onOpenDeviceSpecs: () => void;
  onChangeConfig: (next: InferenceConfig) => void;
  onOpenVisualizer: () => void;
}

const CONTEXT_PRESETS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
const KV_OPTIONS: { id: KVCachePrecision; label: string }[] = [
  { id: "FP16", label: "FP16 (Exact)" },
  { id: "Q8_0", label: "Q8_0 (~50% KV)" },
  { id: "Q4_0", label: "Q4_0 (~28% KV)" },
];

export function ModelCalculatorTab({
  models,
  selectedModel,
  modelDetail,
  isLoadingDetail,
  selectedDevice,
  config,
  onSelectModel,
  onSelectDeviceId,
  onOpenDeviceSpecs,
  onChangeConfig,
  onOpenVisualizer,
}: ModelCalculatorTabProps) {
  const [copiedYaml, setCopiedYaml] = useState(false);

  const vendorGroups = useMemo(() => getHardwareGroupedByVendor(), []);

  const activeModel: HFModelSummary | HFModelDetail =
    modelDetail && modelDetail.id === selectedModel.id ? modelDetail : selectedModel;

  const arch = useMemo(
    () => inferArchitectureFallback(activeModel),
    [activeModel]
  );

  const estimate = useMemo(
    () => calculateInference(activeModel, selectedDevice, config),
    [activeModel, selectedDevice, config]
  );

  const localAiYaml = useMemo(
    () => generateLocalAIYaml(activeModel, config, estimate),
    [activeModel, config, estimate]
  );

  const maxCtxSlider = Math.max(131072, arch.maxContextLength || 131072);

  const handleCopyYaml = async () => {
    try {
      await navigator.clipboard.writeText(localAiYaml);
      setCopiedYaml(true);
      window.setTimeout(() => setCopiedYaml(false), 1800);
    } catch {
      // Ignore clipboard denials in sandboxed browsers
    }
  };

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      {/* LEFT 5 COLS: HUGGING FACE FUZZY PICKER */}
      <section className="surface flex flex-col gap-4 xl:col-span-5">
        <div className="flex shrink-0 items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-bold">
              Hugging Face Model Registry
            </h2>
            <p className="muted text-xs">
              Fuzzy search {models.length} synced models or query the Hub live.
            </p>
          </div>
          {isLoadingDetail && (
            <span className="well inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 font-mono text-[11px] text-[var(--soft-blue)]">
              <Loader2 className="h-3 w-3 animate-spin" />
              Fetching tree
            </span>
          )}
        </div>

        <ModelFuzzyPicker
          models={models}
          selectedModel={selectedModel}
          onSelectModel={onSelectModel}
        />
      </section>

      {/* RIGHT 7 COLS: MODEL SPECS + CONTEXT FIT + 3D LAUNCH + LOCALAI YAML */}
      <section className="space-y-5 xl:col-span-7">
        {/* Top Selected Model Header + Visualize 3D Button */}
        <div className="surface space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-lg font-bold tracking-tight">
                  {activeModel.name}
                </h2>
                <span className="well rounded-lg px-2.5 py-0.5 font-mono text-[11px] font-bold uppercase text-[var(--soft-blue)]">
                  {activeModel.primaryFormat}
                </span>
                <span className="well rounded-lg px-2.5 py-0.5 font-mono text-[11px] font-semibold">
                  {arch.source === "config.json" ? "config.json verified" : arch.source}
                </span>
              </div>
              <a
                href={`https://huggingface.co/${activeModel.id}`}
                target="_blank"
                rel="noreferrer"
                className="muted mt-1 inline-flex items-center gap-1 font-mono text-xs hover:text-[var(--soft-blue)]"
              >
                <span>{activeModel.id}</span>
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>

            <button
              type="button"
              onClick={onOpenVisualizer}
              className="button button--primary inline-flex items-center gap-2 px-4 py-2.5 text-xs font-bold"
            >
              <Box className="h-4 w-4" />
              <span>Visualize 3D Architecture</span>
            </button>
          </div>

          {/* Model Size & Architecture Summary Metrics */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="well rounded-xl p-3">
              <span className="muted block font-mono text-[10px]">
                Parameters
              </span>
              <span className="mt-0.5 block font-mono text-sm font-bold tabular">
                {formatParams(activeModel.parametersB)}
                {activeModel.isMoE
                  ? ` (${formatParams(activeModel.activeParametersB)} act)`
                  : ""}
              </span>
            </div>

            <div className="well rounded-xl p-3">
              <span className="muted block font-mono text-[10px]">
                FP16 Size
              </span>
              <span className="mt-0.5 block font-mono text-sm font-bold tabular">
                {activeModel.exactBaseBytes
                  ? formatBytes(activeModel.exactBaseBytes)
                  : `${activeModel.estimatedDiskGBAtFP16.toFixed(1)} GB`}
              </span>
            </div>

            <div className="well rounded-xl p-3">
              <span className="muted block font-mono text-[10px]">
                Q4_K_M Size
              </span>
              <span className="mt-0.5 block font-mono text-sm font-bold tabular text-[var(--soft-blue)]">
                ~{activeModel.estimatedDiskGBAtQ4.toFixed(1)} GB
              </span>
            </div>

            <div className="well rounded-xl p-3">
              <span className="muted block font-mono text-[10px]">
                Layers / Heads
              </span>
              <span className="mt-0.5 block font-mono text-sm font-bold tabular">
                {arch.numLayers}L · {arch.numAttentionHeads}Q/{arch.numKeyValueHeads}KV
              </span>
            </div>
          </div>
        </div>

        {/* Target Hardware Selector + Inference Tuning Controls */}
        <div className="surface space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4 text-[var(--soft-blue)]" />
              <h3 className="text-sm font-bold">
                Hardware, Quantization & Context Fit Parameters
              </h3>
            </div>

            <button
              type="button"
              onClick={onOpenDeviceSpecs}
              className="button inline-flex items-center gap-1.5 px-3 py-1.5 text-xs"
            >
              <Info className="h-3.5 w-3.5" />
              <span>Device Parameters</span>
            </button>
          </div>

          {/* All GPUs & Apple Silicon Optgroup Select + Apple Unified Ratio */}
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="well space-y-1.5 rounded-xl p-3.5">
              <label
                htmlFor="calc-device-select"
                className="muted flex items-center justify-between font-mono text-[11px] font-semibold"
              >
                <span className="inline-flex items-center gap-1.5">
                  <Cpu className="h-3.5 w-3.5 text-[var(--soft-blue)]" />
                  Target Accelerator
                </span>
                <span className="tabular">ALL {HARDWARE_CATALOG.length}</span>
              </label>
              <select
                id="calc-device-select"
                value={selectedDevice.id}
                onChange={(e) => onSelectDeviceId(e.target.value)}
                className="surface w-full rounded-xl px-3 py-2 text-xs font-semibold focus:outline-none"
              >
                {vendorGroups.map((group) => (
                  <optgroup
                    key={group.label}
                    label={`${group.label} (${group.devices.length})`}
                  >
                    {group.devices.map((dev) => (
                      <option key={dev.id} value={dev.id}>
                        {dev.name} — {dev.vramGB} GB ({dev.bandwidthGBs} GB/s)
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>

            <div className="well space-y-1.5 rounded-xl p-3.5">
              <label
                htmlFor="calc-quant-select"
                className="muted block font-mono text-[11px] font-semibold"
              >
                Quantization Precision
              </label>
              <select
                id="calc-quant-select"
                value={config.quantization}
                onChange={(e) =>
                  onChangeConfig({
                    ...config,
                    quantization: e.target.value as QuantizationLevel,
                  })
                }
                className="surface w-full rounded-xl px-3 py-2 text-xs font-semibold focus:outline-none"
              >
                {QUANTIZATION_SPECS.map((q) => (
                  <option key={q.id} value={q.id}>
                    {q.label} — {q.bpw} bpw ({q.qualityNote})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Context Window Slider + Numeric Input */}
          <div className="well space-y-2.5 rounded-xl p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <label
                htmlFor="calc-context-number"
                className="flex items-center gap-1.5 text-xs font-bold"
              >
                <Layers className="h-3.5 w-3.5 text-[var(--soft-blue)]" />
                Context Window (Tokens)
              </label>
              <div className="flex items-center gap-2">
                <input
                  id="calc-context-number"
                  type="number"
                  min={512}
                  max={maxCtxSlider}
                  step={512}
                  value={config.contextLength}
                  onChange={(e) => {
                    const parsed = Number(e.target.value);
                    if (Number.isFinite(parsed) && parsed >= 256) {
                      onChangeConfig({
                        ...config,
                        contextLength: Math.min(maxCtxSlider, parsed),
                      });
                    }
                  }}
                  className="surface w-28 rounded-xl px-2.5 py-1 text-right font-mono text-xs font-bold tabular focus:outline-none"
                />
                <span className="muted font-mono text-[11px]">
                  max {estimate.maxContextOnDevice.toLocaleString()} fit
                </span>
              </div>
            </div>

            <input
              type="range"
              min={512}
              max={maxCtxSlider}
              step={512}
              value={config.contextLength}
              onChange={(e) =>
                onChangeConfig({
                  ...config,
                  contextLength: Number(e.target.value),
                })
              }
              className="soft-range w-full"
              aria-label="Context window slider"
            />

            <div className="flex flex-wrap gap-1.5">
              {CONTEXT_PRESETS.filter((c) => c <= maxCtxSlider).map((ctx) => (
                <button
                  key={ctx}
                  type="button"
                  onClick={() => onChangeConfig({ ...config, contextLength: ctx })}
                  className={`rounded-lg px-2.5 py-1 font-mono text-[11px] font-semibold ${
                    config.contextLength === ctx
                      ? "button button--primary px-2.5 py-1"
                      : "button button--neutral px-2.5 py-1"
                  }`}
                >
                  {formatCompactNumber(ctx)}
                </button>
              ))}
            </div>
          </div>

          {/* Batch Size, KV Cache, Flash Attention & Apple Unified / Multi-GPU */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="well space-y-1.5 rounded-xl p-3">
              <label
                htmlFor="calc-kv-select"
                className="muted block font-mono text-[11px]"
              >
                KV Cache Precision
              </label>
              <select
                id="calc-kv-select"
                value={config.kvCachePrecision}
                onChange={(e) =>
                  onChangeConfig({
                    ...config,
                    kvCachePrecision: e.target.value as KVCachePrecision,
                  })
                }
                className="surface w-full rounded-xl px-2.5 py-1.5 text-xs font-semibold focus:outline-none"
              >
                {KV_OPTIONS.map((k) => (
                  <option key={k.id} value={k.id}>
                    {k.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="well space-y-1.5 rounded-xl p-3">
              <label
                htmlFor="calc-batch-input"
                className="muted block font-mono text-[11px]"
              >
                Concurrent Batch
              </label>
              <select
                id="calc-batch-input"
                value={config.batchSize}
                onChange={(e) =>
                  onChangeConfig({
                    ...config,
                    batchSize: Number(e.target.value),
                  })
                }
                className="surface w-full rounded-xl px-2.5 py-1.5 text-xs font-semibold focus:outline-none"
              >
                {[1, 2, 4, 8, 16].map((b) => (
                  <option key={b} value={b}>
                    Batch = {b}
                  </option>
                ))}
              </select>
            </div>

            <div className="well flex flex-col justify-between rounded-xl p-3">
              <span className="muted font-mono text-[11px]">FlashAttention-2</span>
              <button
                type="button"
                onClick={() =>
                  onChangeConfig({
                    ...config,
                    flashAttention: !config.flashAttention,
                  })
                }
                className={`mt-1 w-full rounded-xl py-1.5 text-xs font-bold ${
                  config.flashAttention
                    ? "button button--primary py-1.5"
                    : "button button--neutral py-1.5"
                }`}
              >
                {config.flashAttention ? "Enabled" : "Disabled"}
              </button>
            </div>
          </div>

          {selectedDevice.unifiedMemory && (
            <div className="well space-y-1.5 rounded-xl p-3.5">
              <div className="flex items-center justify-between text-xs font-semibold">
                <span>Apple Unified Wired Memory Budget Ratio</span>
                <span className="font-mono tabular text-[var(--soft-blue)]">
                  {Math.round(config.appleWiredRatio * 100)}% (
                  {(selectedDevice.vramGB * config.appleWiredRatio).toFixed(1)} GB)
                </span>
              </div>
              <input
                type="range"
                min={0.5}
                max={selectedDevice.maxUsableRatio}
                step={0.05}
                value={config.appleWiredRatio}
                onChange={(e) =>
                  onChangeConfig({
                    ...config,
                    appleWiredRatio: Number(e.target.value),
                  })
                }
                className="soft-range w-full"
                aria-label="Apple wired memory allocation ratio"
              />
            </div>
          )}
        </div>

        {/* Verdict & VRAM Breakdown Card */}
        <div className="surface space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              {estimate.supported ? (
                <span className="badge--ok inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 font-mono text-xs font-bold">
                  <CheckCircle2 className="h-4 w-4" />
                  FITS INSIDE CONTEXT WINDOW
                </span>
              ) : (
                <span className="badge--warn inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 font-mono text-xs font-bold">
                  <ServerCrash className="h-4 w-4" />
                  DOES NOT FIT ON {selectedDevice.name}
                </span>
              )}
              {estimate.unsupportedReason && (
                <span className="stat--warn font-mono text-xs font-semibold">
                  {estimate.unsupportedReason}
                </span>
              )}
            </div>

            <div className="flex items-center gap-3 font-mono text-xs tabular">
              <span className="inline-flex items-center gap-1 font-bold">
                <Gauge className="h-4 w-4 text-[var(--soft-blue)]" />
                {estimate.estimatedDecodeTps > 0
                  ? `${estimate.estimatedDecodeTps} tok/s decode`
                  : "Image Pipeline"}
              </span>
              {estimate.estimatedPrefillTps > 0 && (
                <span className="muted">
                  ({estimate.estimatedPrefillTps} tok/s prefill)
                </span>
              )}
            </div>
          </div>

          {/* Utilization Meter */}
          <div className="well space-y-2 rounded-xl p-4">
            <div className="flex justify-between font-mono text-xs font-bold tabular">
              <span>
                Required: {estimate.totalRequiredGB.toFixed(2)} GB (Weights{" "}
                {estimate.weightsGB.toFixed(2)} + KV {estimate.kvCacheGB.toFixed(2)} + Act{" "}
                {estimate.overheadGB.toFixed(2)})
              </span>
              <span className={estimate.fits ? "stat--up" : "stat--warn"}>
                {estimate.utilizationPct.toFixed(1)}% of{" "}
                {estimate.effectiveUsableMemoryGB.toFixed(1)} GB usable
              </span>
            </div>
            <div
              className="meter w-full"
              style={
                {
                  "--value": `${Math.min(100, estimate.utilizationPct)}%`,
                  "--meter-color": estimate.fits
                    ? "var(--soft-green)"
                    : "var(--soft-orange)",
                } as React.CSSProperties
              }
            />
          </div>

          {/* LocalAI model.yaml Output */}
          <div className="well space-y-2 rounded-xl p-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold">
                LocalAI Configuration (model.yaml) · {estimate.recommendedBackend}
              </span>
              <button
                type="button"
                onClick={handleCopyYaml}
                className="button button--neutral inline-flex items-center gap-1.5 px-3 py-1.5 text-xs"
              >
                {copiedYaml ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-[var(--soft-green)]" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5" />
                    <span>Copy YAML</span>
                  </>
                )}
              </button>
            </div>
            <pre className="surface soft-scroll overflow-x-auto rounded-xl p-3 font-mono text-[11px] leading-relaxed">
              {localAiYaml}
            </pre>
          </div>
        </div>
      </section>
    </div>
  );
}
