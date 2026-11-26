"use client";

import React, { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  CheckCircle2,
  Cpu,
  HardDrive,
  Info,
  Maximize2,
  Minimize2,
  Search,
  SlidersHorizontal,
  Sparkles,
  X,
  Zap,
} from "lucide-react";
import type {
  HardwareDevice,
  HardwareVendor,
  HFModelSummary,
  InferenceConfig,
  QuantizationLevel,
} from "@/types/inference";
import { HARDWARE_CATALOG } from "@/lib/hardware-catalog";
import {
  filterSupportedModelsForDevice,
  formatCompactNumber,
  formatParams,
  QUANTIZATION_SPECS,
  type CompatibleModelResult,
} from "@/lib/inference-math";

interface HardwareCompatibilityTabProps {
  models: HFModelSummary[];
  selectedDevice: HardwareDevice;
  config: InferenceConfig;
  onSelectDeviceId: (id: string) => void;
  onOpenDeviceSpecs: (id?: string) => void;
  onChangeConfig: (next: InferenceConfig) => void;
  onInspectModelInCalculator: (model: HFModelSummary) => void;
  onVisualizeModel: (model: HFModelSummary) => void;
}

type VendorFilter = "all" | HardwareVendor;
type ModelSortMode = "downloads" | "decode" | "params" | "headroom";

const VENDOR_FILTERS: { id: VendorFilter; label: string }[] = [
  { id: "all", label: "All Silicon" },
  { id: "apple", label: "Apple Silicon" },
  { id: "nvidia-consumer", label: "NVIDIA Consumer" },
  { id: "nvidia-datacenter", label: "NVIDIA Pro / DC" },
  { id: "amd", label: "AMD Radeon / MI" },
  { id: "intel", label: "Intel Arc / Gaudi" },
];

const CONTEXT_CHIPS = [4096, 8192, 16384, 32768, 65536, 131072];
const INITIAL_COMPAT_BATCH = 24;

interface DeviceCardProps {
  device: HardwareDevice;
  isSelected: boolean;
  onSelect: (id: string) => void;
  onOpenSpecs: (id: string) => void;
}

const DeviceDeckCard = React.memo(function DeviceDeckCard({
  device,
  isSelected,
  onSelect,
  onOpenSpecs,
}: DeviceCardProps) {
  return (
    <div
      className={`group flex flex-col justify-between rounded-xl p-3 transition ${
        isSelected
          ? "button button--primary p-3 text-left"
          : "surface p-3 text-left"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <button
          type="button"
          onClick={() => onSelect(device.id)}
          className="min-w-0 flex-1 text-left focus:outline-none"
        >
          <p
            className={`font-mono text-[10px] font-semibold uppercase ${
              isSelected ? "text-white/80" : "muted"
            }`}
          >
            {device.vendorLabel} · {device.architecture}
          </p>
          <h3 className="mt-0.5 truncate text-xs font-bold">{device.name}</h3>
        </button>

        <button
          type="button"
          onClick={() => onOpenSpecs(device.id)}
          aria-label={`Inspect hardware parameters for ${device.name}`}
          className={`shrink-0 rounded-lg p-1.5 transition ${
            isSelected
              ? "bg-white/20 text-white hover:bg-white/30"
              : "well p-1.5 text-[var(--soft-blue)]"
          }`}
        >
          <Info className="h-3.5 w-3.5" />
        </button>
      </div>

      <button
        type="button"
        onClick={() => onSelect(device.id)}
        className="mt-2.5 flex w-full items-center justify-between font-mono text-[11px] tabular focus:outline-none"
      >
        <span className="font-bold">
          {device.vramGB} GB {device.memoryType}
        </span>
        <span className={isSelected ? "text-white/85" : "muted"}>
          {device.bandwidthGBs} GB/s
        </span>
      </button>
    </div>
  );
});

interface CompatibleCardProps {
  entry: CompatibleModelResult;
  onInspect: (model: HFModelSummary) => void;
  onVisualize: (model: HFModelSummary) => void;
}

const CompatibleModelCard = React.memo(function CompatibleModelCard({
  entry,
  onInspect,
  onVisualize,
}: CompatibleCardProps) {
  const { model, estimate } = entry;
  return (
    <div className="surface flex flex-col justify-between space-y-3 rounded-xl p-4">
      <div>
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-1.5">
              <h4 className="truncate text-sm font-bold">{model.name}</h4>
              <span className="well rounded px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase text-[var(--soft-blue)]">
                {model.primaryFormat}
              </span>
              {model.isMoE && (
                <span className="badge--ok rounded px-1.5 py-0.5 font-mono text-[10px] font-bold">
                  MoE
                </span>
              )}
            </div>
            <p className="muted mt-0.5 truncate font-mono text-[11px]">
              {model.id}
            </p>
          </div>

          <span className="badge--ok inline-flex shrink-0 items-center gap-1 rounded-lg px-2 py-0.5 font-mono text-[11px] font-bold tabular">
            <CheckCircle2 className="h-3 w-3" />
            {estimate.utilizationPct.toFixed(0)}%
          </span>
        </div>

        <div className="well mt-3 grid grid-cols-3 gap-2 rounded-xl p-2.5 font-mono text-[11px] tabular">
          <div>
            <span className="muted block text-[10px]">Params</span>
            <span className="font-bold">{formatParams(model.parametersB)}</span>
          </div>
          <div>
            <span className="muted block text-[10px]">Total VRAM</span>
            <span className="font-bold">
              {estimate.totalRequiredGB.toFixed(1)} GB
            </span>
          </div>
          <div>
            <span className="muted block text-[10px]">Speed</span>
            <span className="stat--up font-bold">
              {estimate.estimatedDecodeTps > 0
                ? `${estimate.estimatedDecodeTps} tok/s`
                : "Image"}
            </span>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between gap-2">
        <span className="muted font-mono text-[11px] tabular">
          Headroom: +{estimate.headroomGB.toFixed(1)} GB ·{" "}
          {estimate.recommendedBackend}
        </span>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => onInspect(model)}
            className="button button--neutral px-2.5 py-1.5 text-xs"
          >
            Calculate
          </button>
          <button
            type="button"
            onClick={() => onVisualize(model)}
            className="button button--primary inline-flex items-center gap-1 px-2.5 py-1.5 text-xs"
          >
            <Box className="h-3.5 w-3.5" />
            <span>3D</span>
          </button>
        </div>
      </div>
    </div>
  );
});

export function HardwareCompatibilityTab({
  models,
  selectedDevice,
  config,
  onSelectDeviceId,
  onOpenDeviceSpecs,
  onChangeConfig,
  onInspectModelInCalculator,
  onVisualizeModel,
}: HardwareCompatibilityTabProps) {
  const [vendorFilter, setVendorFilter] = useState<VendorFilter>("all");
  const [hwSearch, setHwSearch] = useState("");
  const [deckExpanded, setDeckExpanded] = useState(false);

  const [modelSearch, setModelSearch] = useState("");
  const [sortMode, setSortMode] = useState<ModelSortMode>("downloads");
  const loadMoreSentinelRef = useRef<HTMLDivElement | null>(null);

  const deferredHwSearch = useDeferredValue(hwSearch);
  const deferredModelSearch = useDeferredValue(modelSearch);
  const deferredConfig = useDeferredValue(config);

  const filterKey = `${selectedDevice.id}:${deferredConfig.quantization}:${deferredConfig.contextLength}:${deferredModelSearch}:${sortMode}`;
  const [limitOverride, setLimitOverride] = useState<{
    key: string;
    limit: number;
  }>({ key: "", limit: INITIAL_COMPAT_BATCH });
  const visibleLimit =
    limitOverride.key === filterKey ? limitOverride.limit : INITIAL_COMPAT_BATCH;

  const filteredDevices = useMemo(() => {
    const q = deferredHwSearch.trim().toLowerCase();
    return HARDWARE_CATALOG.filter((d) => {
      if (vendorFilter !== "all" && d.vendor !== vendorFilter) return false;
      if (!q) return true;
      return `${d.name} ${d.vendorLabel} ${d.architecture} ${d.vramGB}gb ${d.memoryType}`
        .toLowerCase()
        .includes(q);
    });
  }, [vendorFilter, deferredHwSearch]);

  // Strict supported-only filter across ALL catalog models
  const compatibleResults = useMemo(
    () => filterSupportedModelsForDevice(models, selectedDevice, deferredConfig),
    [models, selectedDevice, deferredConfig]
  );

  const sortedCompatible = useMemo(() => {
    const q = deferredModelSearch.trim().toLowerCase();
    const list = q
      ? compatibleResults.filter(({ model }) =>
          `${model.id} ${model.name} ${model.architectureHint} ${model.primaryFormat}`
            .toLowerCase()
            .includes(q)
        )
      : compatibleResults;

    return [...list].sort((a, b) => {
      if (sortMode === "decode") {
        return b.estimate.estimatedDecodeTps - a.estimate.estimatedDecodeTps;
      }
      if (sortMode === "params") {
        return b.model.parametersB - a.model.parametersB;
      }
      if (sortMode === "headroom") {
        return b.estimate.headroomGB - a.estimate.headroomGB;
      }
      return b.model.downloads - a.model.downloads;
    });
  }, [compatibleResults, deferredModelSearch, sortMode]);

  // Progressive reveal of ALL compatible models as the user scrolls
  useEffect(() => {
    const el = loadMoreSentinelRef.current;
    if (!el || visibleLimit >= sortedCompatible.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          setLimitOverride((prev) => {
            const base = prev.key === filterKey ? prev.limit : INITIAL_COMPAT_BATCH;
            return {
              key: filterKey,
              limit: Math.min(sortedCompatible.length, base + INITIAL_COMPAT_BATCH),
            };
          });
        }
      },
      { rootMargin: "200px" }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [visibleLimit, sortedCompatible.length, filterKey]);

  const renderedCompatible = sortedCompatible.slice(0, visibleLimit);

  return (
    <div className="space-y-6">
      {/* SECTION 1: COMPLETE AI HARDWARE DECK (ALL APPLE SILICON + GPUS IN MARKET) */}
      <section className="surface space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="well flex h-10 w-10 items-center justify-center rounded-xl p-0 text-[var(--soft-blue)]">
              <Cpu className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-bold">
                AI Graphics Cards & Apple Silicon Catalog
              </h2>
              <p className="muted text-xs">
                Showing {filteredDevices.length} of {HARDWARE_CATALOG.length}{" "}
                accelerators · Select hardware to filter compatible Hugging Face models, or click Info for full silicon specs.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setDeckExpanded((v) => !v)}
              className="button button--neutral inline-flex items-center gap-1.5 px-3 py-2 text-xs"
            >
              {deckExpanded ? (
                <Minimize2 className="h-3.5 w-3.5" />
              ) : (
                <Maximize2 className="h-3.5 w-3.5" />
              )}
              <span>
                {deckExpanded
                  ? "Compact Grid"
                  : `Expand All ${filteredDevices.length}`}
              </span>
            </button>

            <button
              type="button"
              onClick={() => onOpenDeviceSpecs(selectedDevice.id)}
              className="button inline-flex items-center gap-1.5 px-3 py-2 text-xs"
            >
              <Info className="h-3.5 w-3.5" />
              <span>{selectedDevice.name} Specs</span>
            </button>
          </div>
        </div>

        {/* Vendor Filter Pills + Hardware Search */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap gap-1.5">
            {VENDOR_FILTERS.map((v) => (
              <button
                key={v.id}
                type="button"
                onClick={() => setVendorFilter(v.id)}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                  vendorFilter === v.id
                    ? "button button--primary px-3 py-1.5"
                    : "button button--neutral px-3 py-1.5"
                }`}
              >
                {v.label}
              </button>
            ))}
          </div>

          <div className="well flex w-full items-center gap-2 rounded-xl px-3 py-2 sm:w-72">
            <Search className="h-3.5 w-3.5 text-[var(--soft-blue)]" />
            <input
              type="search"
              value={hwSearch}
              onChange={(e) => setHwSearch(e.target.value)}
              placeholder="Filter GPUs or Apple Silicon..."
              aria-label="Filter hardware accelerators"
              className="w-full bg-transparent text-xs text-[var(--soft-text)] placeholder:text-[var(--soft-muted)] focus:outline-none"
            />
            {hwSearch && (
              <button
                type="button"
                onClick={() => setHwSearch("")}
                aria-label="Clear hardware filter"
                className="muted p-0.5 hover:text-[var(--soft-text)]"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* All Matching Market GPUs & Apple Silicon */}
        <div
          className={`well soft-scroll grid grid-cols-1 gap-3 rounded-xl p-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 ${
            deckExpanded ? "max-h-none" : "max-h-[340px] overflow-y-auto"
          }`}
        >
          {filteredDevices.map((dev) => (
            <DeviceDeckCard
              key={dev.id}
              device={dev}
              isSelected={dev.id === selectedDevice.id}
              onSelect={onSelectDeviceId}
              onOpenSpecs={(id) => onOpenDeviceSpecs(id)}
            />
          ))}
        </div>
      </section>

      {/* SECTION 2: STRICT SUPPORTED-ONLY HUGGING FACE MODELS */}
      <section className="surface space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="well flex h-10 w-10 items-center justify-center rounded-xl p-0 text-[var(--soft-green)]">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-bold">
                Supported Models for {selectedDevice.name}
              </h2>
              <p className="muted text-xs tabular">
                <span className="stat--up font-bold">
                  {sortedCompatible.length} supported
                </span>{" "}
                of {models.length} Hugging Face models ·{" "}
                {models.length - compatibleResults.length} incompatible or OOM models hidden
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Multi-GPU Control when hardware supports it */}
            {selectedDevice.supportsMultiGpu && (
              <div className="well flex items-center gap-2 rounded-xl px-3 py-1.5 text-xs">
                <span className="muted font-mono text-[11px]">GPUs:</span>
                <select
                  value={config.gpuCount}
                  onChange={(e) =>
                    onChangeConfig({
                      ...config,
                      gpuCount: Number(e.target.value),
                    })
                  }
                  aria-label="GPU count for Hardware Fit compatibility"
                  className="bg-transparent font-mono font-bold focus:outline-none"
                >
                  {[1, 2, 4, 8]
                    .filter((n) => n <= selectedDevice.maxGpuCount)
                    .map((n) => (
                      <option key={n} value={n}>
                        {n}x
                      </option>
                    ))}
                </select>
              </div>
            )}

            {/* Quantization Selector */}
            <div className="well flex items-center gap-2 rounded-xl px-3 py-1.5 text-xs">
              <SlidersHorizontal className="h-3.5 w-3.5 text-[var(--soft-blue)]" />
              <select
                value={config.quantization}
                onChange={(e) =>
                  onChangeConfig({
                    ...config,
                    quantization: e.target.value as QuantizationLevel,
                  })
                }
                aria-label="Target quantization precision"
                className="bg-transparent font-mono font-bold focus:outline-none"
              >
                {QUANTIZATION_SPECS.map((q) => (
                  <option key={q.id} value={q.id}>
                    {q.id} ({q.bpw} bpw)
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Context Window Scrubber + Compatible Search & Sort */}
        <div className="well grid grid-cols-1 gap-4 rounded-xl p-4 lg:grid-cols-12">
          <div className="space-y-2 lg:col-span-6">
            <div className="flex items-center justify-between text-xs font-semibold">
              <span className="inline-flex items-center gap-1.5">
                <HardDrive className="h-3.5 w-3.5 text-[var(--soft-blue)]" />
                Context Window Filter
              </span>
              <span className="font-mono tabular text-[var(--soft-blue)]">
                {config.contextLength.toLocaleString()} tokens
              </span>
            </div>
            <input
              type="range"
              min={512}
              max={131072}
              step={512}
              value={Math.min(131072, config.contextLength)}
              onChange={(e) =>
                onChangeConfig({
                  ...config,
                  contextLength: Number(e.target.value),
                })
              }
              className="soft-range w-full"
              aria-label="Context window tokens filter"
            />
            <div className="flex flex-wrap gap-1.5">
              {CONTEXT_CHIPS.map((ctx) => (
                <button
                  key={ctx}
                  type="button"
                  onClick={() => onChangeConfig({ ...config, contextLength: ctx })}
                  className={`rounded-lg px-2 py-1 font-mono text-[11px] ${
                    config.contextLength === ctx
                      ? "button button--primary px-2 py-1"
                      : "button button--neutral px-2 py-1"
                  }`}
                >
                  {formatCompactNumber(ctx)}
                </button>
              ))}
            </div>
          </div>

          <div className="flex flex-col justify-between gap-2 lg:col-span-6">
            <div className="flex flex-wrap items-center gap-2">
              <div className="surface flex flex-1 items-center gap-2 rounded-xl px-3 py-2">
                <Search className="h-3.5 w-3.5 text-[var(--soft-blue)]" />
                <input
                  type="search"
                  value={modelSearch}
                  onChange={(e) => setModelSearch(e.target.value)}
                  placeholder="Filter supported models..."
                  aria-label="Filter compatible Hugging Face models"
                  className="w-full bg-transparent text-xs text-[var(--soft-text)] placeholder:text-[var(--soft-muted)] focus:outline-none"
                />
              </div>

              <div className="surface flex items-center gap-1.5 rounded-xl px-3 py-2 text-xs">
                <Zap className="h-3.5 w-3.5 text-[var(--soft-orange)]" />
                <select
                  value={sortMode}
                  onChange={(e) => setSortMode(e.target.value as ModelSortMode)}
                  aria-label="Sort compatible models"
                  className="bg-transparent font-mono font-semibold focus:outline-none"
                >
                  <option value="downloads">Sort: Downloads</option>
                  <option value="decode">Sort: Fastest tok/s</option>
                  <option value="params">Sort: Largest Params</option>
                  <option value="headroom">Sort: Most Headroom</option>
                </select>
              </div>
            </div>

            <p className="muted font-mono text-[11px]">
              Showing {renderedCompatible.length} of {sortedCompatible.length} compatible models at {config.quantization} ·{" "}
              {config.contextLength.toLocaleString()} ctx
            </p>
          </div>
        </div>

        {/* Progressive Compatible Model Cards Grid */}
        {sortedCompatible.length === 0 ? (
          <div className="well rounded-xl p-8 text-center">
            <p className="text-sm font-bold">
              No models fit inside {selectedDevice.name} at {config.quantization} and{" "}
              {config.contextLength.toLocaleString()} context
            </p>
            <p className="muted mt-1 text-xs">
              Try selecting a smaller quantization (Q4_K_M / IQ4_XS), reducing the context window, or enabling multiple GPUs.
            </p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 gap-3.5 md:grid-cols-2 xl:grid-cols-3">
              {renderedCompatible.map((entry) => (
                <CompatibleModelCard
                  key={entry.model.id}
                  entry={entry}
                  onInspect={onInspectModelInCalculator}
                  onVisualize={onVisualizeModel}
                />
              ))}
            </div>

            {renderedCompatible.length < sortedCompatible.length && (
              <div
                ref={loadMoreSentinelRef}
                className="flex flex-col items-center justify-center gap-2 pt-2"
              >
                <button
                  type="button"
                  onClick={() =>
                    setLimitOverride({
                      key: filterKey,
                      limit: sortedCompatible.length,
                    })
                  }
                  className="button px-4 py-2 text-xs"
                >
                  Show All Remaining{" "}
                  {sortedCompatible.length - renderedCompatible.length} Compatible Models
                </button>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
