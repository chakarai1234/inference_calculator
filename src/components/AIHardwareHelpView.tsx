"use client";

import React, { useDeferredValue, useMemo, useState } from "react";
import {
  BookOpen,
  Calculator,
  ChevronDown,
  ChevronUp,
  Cpu,
  HardDrive,
  Layers,
  RotateCcw,
  Search,
  Sparkles,
  Zap,
} from "lucide-react";
import { getHardwareGroupedByVendor } from "@/lib/hardware-catalog";
import {
  buildContextScaleSweep,
  buildParameterScaleSweep,
  computeFormulaLab,
  DEFAULT_FORMULA_LAB_INPUT,
  formatContextShort,
  formatParamScaleShort,
  LAB_ATTENTION_LABELS,
  LAB_KV_BYTES,
  LAB_QUANT_PRESETS,
  LAB_SCALE_PRESETS,
  type FormulaLabInput,
  type LabAttentionMode,
  type LabKVPrecision,
} from "@/lib/formula-lab";

type HelpCategory = "all" | "gpu" | "ram" | "cpu" | "context";
type SweepMode = "params" | "context";

interface ParameterGuideEntry {
  id: string;
  category: Exclude<HelpCategory, "all">;
  categoryLabel: string;
  title: string;
  impact: string;
  summary: string;
  whyAI: string;
  formulaOrRule: string;
}

const GUIDE_ENTRIES: ParameterGuideEntry[] = [
  {
    id: "gpu-bandwidth",
    category: "gpu",
    categoryLabel: "GPU",
    title: "Memory Bandwidth (GB/s)",
    impact: "Governs Decode tok/s",
    summary:
      "How many gigabytes per second the GPU (or Apple Silicon SoC) can stream from VRAM/Unified Memory into compute cores.",
    whyAI:
      "Autoregressive LLM text generation reads every active parameter for every single output token. Bandwidth—not TFLOPS—sets your tokens/second speed.",
    formulaOrRule: "Decode tok/s ≈ (Memory Bandwidth × 0.76) ÷ Active Model Size (GB)",
  },
  {
    id: "gpu-tflops",
    category: "gpu",
    categoryLabel: "GPU",
    title: "FP16 / FP8 / INT4 Compute (TFLOPS & TOPS)",
    impact: "Governs Prompt Prefill",
    summary:
      "Peak matrix multiplication throughput of CUDA/Tensor/XMX/Metal cores across precisions.",
    whyAI:
      "When you paste a long 16K–128K prompt or RAG document, the GPU processes all input tokens in parallel (compute-bound). Higher TFLOPS dramatically cuts Time-To-First-Token (TTFT).",
    formulaOrRule: "Prefill tok/s ≈ Effective TFLOPS ÷ (2 × Active Parameters B)",
  },
  {
    id: "gpu-interconnect",
    category: "gpu",
    categoryLabel: "GPU",
    title: "Interconnect (NVLink / UltraFusion / PCIe)",
    impact: "Multi-GPU Scaling",
    summary:
      "High-speed link connecting multiple GPUs or chiplets (NVLink 600–1800 GB/s, UltraFusion 2.5 TB/s, PCIe 5.0 64 GB/s).",
    whyAI:
      "When tensor-splitting a 70B–2T model across 2–16 GPUs, layers synchronize activations at every block. Slow PCIe bottlenecks scaling; NVLink keeps multi-GPU speed crisp.",
    formulaOrRule: "Tensor Parallel efficiency: ~0.92x/GPU on NVLink vs ~0.78x on PCIe",
  },
  {
    id: "gpu-tdp",
    category: "gpu",
    categoryLabel: "GPU",
    title: "TDP & Sustained Thermal Power (Watts)",
    impact: "Efficiency & Acoustics",
    summary:
      "Maximum sustained power draw and heat dissipation under continuous inference load.",
    whyAI:
      "Apple Silicon & RTX L4 run 30W–150W quietly at desk; datacenter H200/B200 pull 700W–1000W per chip requiring liquid or server rack airflow.",
    formulaOrRule: "Energy/token = GPU Watts ÷ Sustained tok/s",
  },
  {
    id: "ram-vram-capacity",
    category: "ram",
    categoryLabel: "RAM & VRAM",
    title: "VRAM vs Unified Memory Capacity (GB)",
    impact: "Governs Which Models Fit",
    summary:
      "Dedicated discrete memory on PCIe GPUs (8–288 GB) vs shared CPU+GPU memory pool on Apple Silicon, Strix Halo, and Grace Blackwell.",
    whyAI:
      "Model weights, KV cache, and workspace buffers MUST fit inside fast GPU-addressable memory to run at full speed. Spilling to system DDR over PCIe drops speed 10x–30x.",
    formulaOrRule: "Usable AI Pool: ~95% on discrete GPUs; 75% default (90% sysctl max) on macOS",
  },
  {
    id: "ram-memory-type",
    category: "ram",
    categoryLabel: "RAM & VRAM",
    title: "Memory Tech & Bus Width (HBM3e / GDDR7 / LPDDR5X)",
    impact: "Determines Bandwidth",
    summary:
      "Physical DRAM silicon and interface width (128-bit to 8192-bit wide HBM stacks).",
    whyAI:
      "Bandwidth = Memory Clock × Bus Width. HBM3e achieves 4.8–8.0 TB/s via ultra-wide 6144–8192-bit stacks; GDDR7 hits 1.79 TB/s on 512-bit; LPDDR5X balances huge capacity (128–512 GB) at lower power.",
    formulaOrRule: "RTX 5090 512-bit GDDR7 (1,792 GB/s) vs M3 Ultra 1024-bit LPDDR5 (819 GB/s)",
  },
  {
    id: "ram-host-system",
    category: "ram",
    categoryLabel: "RAM & VRAM",
    title: "Host System RAM & Memory Channels (DDR4/DDR5)",
    impact: "Model Loading & CPU Offload",
    summary: "Motherboard system memory paired with discrete GPUs.",
    whyAI:
      "LocalAI uses mmap to page weights from disk through host RAM into VRAM. Too little host RAM slows cold loading or crashes multi-model swapping and CPU-offloaded MoE layers.",
    formulaOrRule: "Rule of thumb: Host RAM ≥ 1.5× largest model disk file size",
  },
  {
    id: "cpu-channels",
    category: "cpu",
    categoryLabel: "CPU",
    title: "CPU Memory Channels & NUMA Topology",
    impact: "CPU / Partial-Offload Speed",
    summary:
      "Number of independent DDR5 memory lanes feeding the CPU socket (2 channels desktop, 8–12 channels EPYC/Xeon).",
    whyAI:
      "If any layers of a huge MoE model offload to CPU RAM (`n-gpu-layers < max`), token speed drops directly to your CPU's memory channel bandwidth (~70–90 GB/s dual-channel DDR5).",
    formulaOrRule: "Dual-channel DDR5 ≈ 90 GB/s vs 12-channel EPYC ≈ 460 GB/s",
  },
  {
    id: "cpu-simd",
    category: "cpu",
    categoryLabel: "CPU",
    title: "Vector & Matrix ISA (AVX-512, AMX, ARM SME / Neon)",
    impact: "Prompt Ingestion on CPU",
    summary:
      "Hardware matrix and wide SIMD instructions inside modern CPUs.",
    whyAI:
      "llama.cpp compiles specialized kernels using AVX-512/AMX on x64 and DotProd/I8MM/SME on ARM64 to accelerate quantized Q4/Q8 dot products when running on CPU.",
    formulaOrRule: "Apple M4/M5 SME boosts CPU-side prompt prefill >2x vs standard Neon",
  },
  {
    id: "cpu-pcie",
    category: "cpu",
    categoryLabel: "CPU",
    title: "PCIe Generation & Lanes (PCIe 4.0 vs 5.0 x16)",
    impact: "Cold Load & Split Offload",
    summary:
      "Host-to-GPU expansion bus bandwidth (32 GB/s PCIe 4.0 x16, 64 GB/s PCIe 5.0 x16).",
    whyAI:
      "Controls how fast LocalAI copies a 40 GB+ model from NVMe SSD into GPU VRAM and caps communication speed if a workload spills over PCIe.",
    formulaOrRule: "Never run inference GPUs on x1/x4 mining risers for multi-GPU tensor parallel",
  },
  {
    id: "ctx-window",
    category: "context",
    categoryLabel: "Context & Quant",
    title: "Context Window (n_ctx) & KV Cache Scaling",
    impact: "Memory Growth & Long Prompts",
    summary:
      "Maximum conversation history (prompt + RAG docs + generated tokens) kept active in the model's attention memory.",
    whyAI:
      "Even when a 4-bit model weighs only 5 GB, cranking Context Window from 4K to 1M can explode KV Cache memory past 100 GB unless GQA/MLA and KV quantization are used!",
    formulaOrRule: "KV Bytes = 2 × Layers × KV_Heads × Head_Dim × Context × Batch × KV_Bytes",
  },
  {
    id: "ctx-gqa",
    category: "context",
    categoryLabel: "Context & Quant",
    title: "Grouped-Query Attention (GQA) & DeepSeek MLA",
    impact: "4x–20x Context Memory Savings",
    summary:
      "Sharing Key/Value heads (GQA 4:1 / 8:1) or compressing KV into low-rank latents (Multi-Head Latent Attention).",
    whyAI:
      "DeepSeek MLA stores only ~576 compressed elements per layer-token instead of full K/V heads, letting 671B+ models hold 128K context in a fraction of normal KV VRAM.",
    formulaOrRule: "MLA saves >90% KV cache vs MHA and ~4x–8x vs standard GQA",
  },
  {
    id: "ctx-quant",
    category: "context",
    categoryLabel: "Context & Quant",
    title: "Quantization (FP32 → BitNet 1.58b) & Active MoE",
    impact: "Size + Decode Speedup",
    summary:
      "Compressing each weight parameter from 32/16 bits down to 4.85 bits (Q4_K_M), 2-bit IQ2, or ternary 1.58-bit.",
    whyAI:
      "MoE stores Total_Params weights in VRAM, but streams only Active_Params per generated token—making a 671B/37B or 1T/64B MoE run 10x faster than a dense equivalent.",
    formulaOrRule: "Weights GB = (Total_Params_B × Bits_Per_Weight) ÷ 8 ÷ 1.0737",
  },
];

const CATEGORIES: { id: HelpCategory; label: string }[] = [
  { id: "all", label: "All Parameters" },
  { id: "gpu", label: "GPU Compute" },
  { id: "ram", label: "RAM & VRAM" },
  { id: "cpu", label: "CPU & PCIe Bus" },
  { id: "context", label: "Context & Quantization" },
];

const QUICK_PARAM_CHIPS = [0.5, 3, 8, 14, 32, 70, 235, 405, 671, 1000, 2000];
const QUICK_CTX_CHIPS = [2048, 8192, 32768, 131072, 524288, 1048576, 2097152];
const QUICK_VRAM_CHIPS = [8, 16, 24, 48, 64, 80, 128, 192, 256, 512];
const QUICK_GPU_COUNTS = [1, 2, 4, 8, 16];

// Logarithmic slider helpers [0..1000] <-> [min..max]
const PARAM_MIN_LOG = Math.log10(0.1);
const PARAM_MAX_LOG = Math.log10(2000);
const CTX_MIN_LOG = Math.log10(512);
const CTX_MAX_LOG = Math.log10(2097152);

function paramsToSlider(p: number): number {
  const safe = Math.min(2000, Math.max(0.1, Number.isFinite(p) ? p : 8));
  return Math.round(((Math.log10(safe) - PARAM_MIN_LOG) / (PARAM_MAX_LOG - PARAM_MIN_LOG)) * 1000);
}

function sliderToParams(val: number): number {
  const frac = Math.min(1, Math.max(0, val / 1000));
  const raw = Math.pow(10, PARAM_MIN_LOG + frac * (PARAM_MAX_LOG - PARAM_MIN_LOG));
  return raw < 1 ? Number(raw.toFixed(2)) : raw < 20 ? Number(raw.toFixed(1)) : Math.round(raw);
}

function ctxToSlider(ctx: number): number {
  const safe = Math.min(2097152, Math.max(512, Number.isFinite(ctx) ? ctx : 8192));
  return Math.round(((Math.log10(safe) - CTX_MIN_LOG) / (CTX_MAX_LOG - CTX_MIN_LOG)) * 1000);
}

function sliderToCtx(val: number): number {
  const frac = Math.min(1, Math.max(0, val / 1000));
  const raw = Math.pow(10, CTX_MIN_LOG + frac * (CTX_MAX_LOG - CTX_MIN_LOG));
  return Math.max(256, Math.round(raw / 256) * 256);
}

export function AIHardwareHelpView() {
  const [category, setCategory] = useState<HelpCategory>("all");
  const [search, setSearch] = useState("");
  const [lab, setLab] = useState<FormulaLabInput>(DEFAULT_FORMULA_LAB_INPUT);
  const [activePresetId, setActivePresetId] = useState<string>("8b-desktop");
  const [selectedDevicePresetId, setSelectedDevicePresetId] = useState<string>("");
  const [showArchOverrides, setShowArchOverrides] = useState(false);
  const [sweepMode, setSweepMode] = useState<SweepMode>("params");

  const vendorGroups = useMemo(() => getHardwareGroupedByVendor(), []);
  const deferredLab = useDeferredValue(lab);
  const result = useMemo(() => computeFormulaLab(lab), [lab]);
  const paramSweep = useMemo(() => buildParameterScaleSweep(deferredLab), [deferredLab]);
  const contextSweep = useMemo(() => buildContextScaleSweep(deferredLab), [deferredLab]);

  const applyPreset = (presetId: string) => {
    const preset = LAB_SCALE_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    setActivePresetId(presetId);
    setSelectedDevicePresetId("");
    setLab((prev) => ({
      ...prev,
      ...preset.input,
      overrideLayers: undefined,
      overrideHiddenSize: undefined,
      overrideKvHeads: undefined,
      overrideHeadDim: undefined,
    }));
  };

  const applyHardwareDevice = (deviceId: string) => {
    setSelectedDevicePresetId(deviceId);
    if (!deviceId) return;
    for (const grp of vendorGroups) {
      const found = grp.devices.find((d) => d.id === deviceId);
      if (found) {
        setLab((prev) => ({
          ...prev,
          vramPerGpuGB: found.vramGB,
          bandwidthGBs: found.bandwidthGBs,
          fp16TflopsPerGpu: found.fp16Tflops,
          usableMemoryRatio: found.unifiedMemory ? found.defaultUsableRatio : 0.94,
        }));
        break;
      }
    }
  };

  const updateParamsB = (nextTotal: number) => {
    setActivePresetId("");
    setLab((prev) => {
      const safeTotal = Number.isFinite(nextTotal) ? Math.max(0.01, nextTotal) : prev.totalParamsB;
      const ratio =
        prev.isMoE && prev.totalParamsB > 0
          ? Math.min(1, prev.activeParamsB / prev.totalParamsB)
          : 1;
      return {
        ...prev,
        totalParamsB: safeTotal,
        activeParamsB: prev.isMoE
          ? Number(Math.max(0.05, Math.min(safeTotal, safeTotal * ratio)).toFixed(2))
          : safeTotal,
      };
    });
  };

  const filteredEntries = useMemo(() => {
    const q = search.trim().toLowerCase();
    return GUIDE_ENTRIES.filter((item) => {
      if (category !== "all" && item.category !== category) return false;
      if (!q) return true;
      return (
        item.title.toLowerCase().includes(q) ||
        item.summary.toLowerCase().includes(q) ||
        item.whyAI.toLowerCase().includes(q) ||
        item.categoryLabel.toLowerCase().includes(q)
      );
    });
  }, [category, search]);

  return (
    <div className="space-y-6">
      {/* MAIN SECTION: FULL-RANGE INTERACTIVE CONTEXT WINDOW & MEMORY FORMULA LAB */}
      <section className="surface p-5 sm:p-6">
        <div className="flex flex-col justify-between gap-4 border-b border-[var(--soft-border)] pb-4 lg:flex-row lg:items-center">
          <div className="flex items-start gap-3">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-[var(--soft-blue-fill)] text-white">
              <Calculator className="h-5 w-5" />
            </div>
            <div>
              <span className="font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[var(--soft-blue)]">
                Sub-Billion to Multi-Trillion Scale Simulator
              </span>
              <h1 className="text-lg font-bold text-[var(--soft-text)] sm:text-xl">
                Interactive Context Window &amp; Memory Formula Lab
              </h1>
              <p className="muted mt-0.5 text-xs">
                Dial any parameter count (0.1B → 10,000B), custom quantization bits, MoE active params, GQA/MLA attention, and 128–10M token context windows.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`rounded-xl px-3.5 py-2 font-mono text-xs font-bold tabular ${
                result.fits ? "badge--ok" : "badge--warn"
              }`}
            >
              {result.fits
                ? `FITS ON ${result.usablePoolGB.toFixed(1)} GB (${result.utilizationPct.toFixed(0)}% USED)`
                : `OVERFLOWS ${result.usablePoolGB.toFixed(1)} GB · NEED ${result.minGpusNeeded}× ${result.sanitized.vramPerGpuGB}GB`}
            </span>
          </div>
        </div>

        {/* Scale Presets Ribbon (0.5B Edge -> 671B DeepSeek MLA -> 2000B 2T Mega) */}
        <div className="mt-4">
          <p className="muted mb-2 font-mono text-[10px] font-bold uppercase tracking-wider">
            Model &amp; Cluster Presets (0.5B → 2,000B / 2T)
          </p>
          <div className="flex flex-wrap gap-1.5">
            {LAB_SCALE_PRESETS.map((preset) => {
              const active = activePresetId === preset.id;
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => applyPreset(preset.id)}
                  className={`button inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 font-mono text-[11px] ${
                    active ? "button--primary" : "button--neutral"
                  }`}
                >
                  <span>{preset.label}</span>
                  <span
                    className={`rounded px-1.5 py-0.5 text-[9px] font-bold ${
                      active
                        ? "bg-white/20 text-white"
                        : "bg-[var(--soft-shadow)]/50 text-[var(--soft-muted)]"
                    }`}
                  >
                    {preset.badge}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Controls Grid: 4 Soft UI Wells (max 1 nesting level: .surface > .well) */}
        <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Well 1: Model Parameters, MoE & Quantization */}
          <div className="well space-y-4 rounded-2xl p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <label
                htmlFor="lab-params-number"
                className="font-mono text-xs font-bold text-[var(--soft-text)]"
              >
                Model Parameters: {formatParamScaleShort(result.sanitized.totalParamsB)} ({result.sanitized.totalParamsB}B)
              </label>

              <div className="flex items-center gap-2">
                <input
                  id="lab-params-number"
                  type="number"
                  min={0.05}
                  max={10000}
                  step="any"
                  value={lab.totalParamsB}
                  onChange={(e) => updateParamsB(parseFloat(e.target.value))}
                  aria-label="Custom billions of parameters"
                  className="surface w-24 rounded-xl px-2.5 py-1 text-right font-mono text-xs font-bold tabular"
                />
                <span className="muted font-mono text-[11px] font-bold">B</span>
              </div>
            </div>

            <input
              type="range"
              min={0}
              max={1000}
              step={1}
              value={paramsToSlider(lab.totalParamsB)}
              onChange={(e) => updateParamsB(sliderToParams(Number(e.target.value)))}
              className="soft-range w-full"
              aria-label="Logarithmic parameter scale slider from 0.1B to 2000B"
            />

            <div className="flex flex-wrap gap-1.5">
              {QUICK_PARAM_CHIPS.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => updateParamsB(p)}
                  className={`button rounded-lg px-2 py-1 font-mono text-[10px] ${
                    Math.abs(lab.totalParamsB - p) < 0.01
                      ? "button--primary px-2 py-1"
                      : "button--neutral px-2 py-1"
                  }`}
                >
                  {formatParamScaleShort(p)}
                </button>
              ))}
            </div>

            {/* Dense vs MoE Switch + Active Params */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[var(--soft-border)] pt-3">
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() =>
                    setLab((prev) => ({
                      ...prev,
                      isMoE: false,
                      activeParamsB: prev.totalParamsB,
                    }))
                  }
                  className={`button rounded-lg px-3 py-1.5 font-mono text-[11px] ${
                    !lab.isMoE ? "button--primary px-3 py-1.5" : "button--neutral px-3 py-1.5"
                  }`}
                >
                  Dense Model
                </button>
                <button
                  type="button"
                  onClick={() =>
                    setLab((prev) => ({
                      ...prev,
                      isMoE: true,
                      activeParamsB:
                        prev.activeParamsB > 0 && prev.activeParamsB < prev.totalParamsB
                          ? prev.activeParamsB
                          : Number(Math.max(0.1, prev.totalParamsB * 0.12).toFixed(1)),
                    }))
                  }
                  className={`button rounded-lg px-3 py-1.5 font-mono text-[11px] ${
                    lab.isMoE ? "button--primary px-3 py-1.5" : "button--neutral px-3 py-1.5"
                  }`}
                >
                  Sparse MoE
                </button>
              </div>

              {lab.isMoE && (
                <div className="flex items-center gap-2">
                  <label
                    htmlFor="lab-active-params"
                    className="muted font-mono text-[11px]"
                  >
                    Active/tok:
                  </label>
                  <input
                    id="lab-active-params"
                    type="number"
                    min={0.05}
                    max={result.sanitized.totalParamsB}
                    step="any"
                    value={lab.activeParamsB}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      if (Number.isFinite(val)) {
                        setLab((prev) => ({ ...prev, activeParamsB: val }));
                      }
                    }}
                    className="surface w-20 rounded-xl px-2 py-1 text-right font-mono text-xs font-bold tabular"
                  />
                  <span className="muted font-mono text-[11px]">
                    B (
                    {Math.round(
                      (result.sanitized.activeParamsB / result.sanitized.totalParamsB) *
                        100
                    )}
                    %)
                  </span>
                </div>
              )}
            </div>

            {/* Quantization Format & Custom Bits-Per-Weight */}
            <div className="grid grid-cols-1 gap-3 border-t border-[var(--soft-border)] pt-3 sm:grid-cols-12">
              <div className="sm:col-span-8">
                <label
                  htmlFor="lab-quant-select"
                  className="muted mb-1 block font-mono text-[10px] font-bold uppercase"
                >
                  Weight Precision Format
                </label>
                <select
                  id="lab-quant-select"
                  value={
                    LAB_QUANT_PRESETS.some((q) => q.label === lab.quantLabel)
                      ? lab.quantLabel
                      : "custom"
                  }
                  onChange={(e) => {
                    const found = LAB_QUANT_PRESETS.find((q) => q.label === e.target.value);
                    if (found) {
                      setLab((prev) => ({
                        ...prev,
                        quantLabel: found.label,
                        bitsPerWeight: found.bpw,
                      }));
                    }
                  }}
                  className="surface w-full rounded-xl px-3 py-2 font-mono text-xs font-semibold"
                >
                  {LAB_QUANT_PRESETS.map((q) => (
                    <option key={q.id} value={q.label}>
                      {q.label} ({q.bpw} bpw) — {q.note}
                    </option>
                  ))}
                  <option value="custom">Custom Bits Per Weight ({lab.bitsPerWeight} bpw)</option>
                </select>
              </div>

              <div className="sm:col-span-4">
                <label
                  htmlFor="lab-custom-bpw"
                  className="muted mb-1 block font-mono text-[10px] font-bold uppercase"
                >
                  Bits / Weight
                </label>
                <input
                  id="lab-custom-bpw"
                  type="number"
                  min={0.5}
                  max={32}
                  step={0.05}
                  value={lab.bitsPerWeight}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (Number.isFinite(val)) {
                      setLab((prev) => ({
                        ...prev,
                        quantLabel: "Custom",
                        bitsPerWeight: val,
                      }));
                    }
                  }}
                  className="surface w-full rounded-xl px-3 py-2 text-right font-mono text-xs font-bold tabular"
                />
              </div>
            </div>
          </div>

          {/* Well 2: Context Window, Batch, Attention Mode & Architecture Overrides */}
          <div className="well space-y-4 rounded-2xl p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <label
                htmlFor="lab-context-number"
                className="font-mono text-xs font-bold text-[var(--soft-text)]"
              >
                Context Window: {formatContextShort(result.sanitized.contextLength)} tokens
              </label>

              <div className="flex items-center gap-2">
                <input
                  id="lab-context-number"
                  type="number"
                  min={128}
                  max={10000000}
                  step={256}
                  value={lab.contextLength}
                  onChange={(e) => {
                    const val = parseInt(e.target.value, 10);
                    if (Number.isFinite(val)) {
                      setLab((prev) => ({ ...prev, contextLength: val }));
                    }
                  }}
                  aria-label="Custom context window length"
                  className="surface w-28 rounded-xl px-2.5 py-1 text-right font-mono text-xs font-bold tabular"
                />
                <span className="muted font-mono text-[11px]">tok</span>
              </div>
            </div>

            <input
              type="range"
              min={0}
              max={1000}
              step={1}
              value={ctxToSlider(lab.contextLength)}
              onChange={(e) =>
                setLab((prev) => ({
                  ...prev,
                  contextLength: sliderToCtx(Number(e.target.value)),
                }))
              }
              className="soft-range w-full"
              aria-label="Logarithmic context length slider from 512 to 2M"
            />

            <div className="flex flex-wrap gap-1.5">
              {QUICK_CTX_CHIPS.map((ctx) => (
                <button
                  key={ctx}
                  type="button"
                  onClick={() => setLab((prev) => ({ ...prev, contextLength: ctx }))}
                  className={`button rounded-lg px-2 py-1 font-mono text-[10px] ${
                    lab.contextLength === ctx
                      ? "button--primary px-2 py-1"
                      : "button--neutral px-2 py-1"
                  }`}
                >
                  {formatContextShort(ctx)}
                </button>
              ))}
            </div>

            {/* Attention Mechanism, KV Precision, Batch Size & FlashAttention */}
            <div className="grid grid-cols-2 gap-3 border-t border-[var(--soft-border)] pt-3 sm:grid-cols-4">
              <div className="col-span-2 sm:col-span-1">
                <label
                  htmlFor="lab-attn-select"
                  className="muted mb-1 block font-mono text-[10px] font-bold uppercase"
                >
                  Attention
                </label>
                <select
                  id="lab-attn-select"
                  value={lab.attentionMode}
                  onChange={(e) =>
                    setLab((prev) => ({
                      ...prev,
                      attentionMode: e.target.value as LabAttentionMode,
                    }))
                  }
                  className="surface w-full rounded-xl px-2.5 py-2 font-mono text-[11px] font-semibold"
                >
                  {(Object.keys(LAB_ATTENTION_LABELS) as LabAttentionMode[]).map((k) => (
                    <option key={k} value={k}>
                      {LAB_ATTENTION_LABELS[k]}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label
                  htmlFor="lab-kv-select"
                  className="muted mb-1 block font-mono text-[10px] font-bold uppercase"
                >
                  KV Cache
                </label>
                <select
                  id="lab-kv-select"
                  value={lab.kvPrecision}
                  onChange={(e) =>
                    setLab((prev) => ({
                      ...prev,
                      kvPrecision: e.target.value as LabKVPrecision,
                    }))
                  }
                  className="surface w-full rounded-xl px-2.5 py-2 font-mono text-[11px] font-semibold"
                >
                  {(Object.keys(LAB_KV_BYTES) as LabKVPrecision[]).map((k) => (
                    <option key={k} value={k}>
                      {k} ({LAB_KV_BYTES[k]}B)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label
                  htmlFor="lab-batch-num"
                  className="muted mb-1 block font-mono text-[10px] font-bold uppercase"
                >
                  Batch
                </label>
                <input
                  id="lab-batch-num"
                  type="number"
                  min={1}
                  max={256}
                  step={1}
                  value={lab.batchSize}
                  onChange={(e) => {
                    const val = parseInt(e.target.value, 10);
                    if (Number.isFinite(val)) {
                      setLab((prev) => ({ ...prev, batchSize: val }));
                    }
                  }}
                  className="surface w-full rounded-xl px-2.5 py-2 text-right font-mono text-xs font-bold tabular"
                />
              </div>

              <div className="flex flex-col justify-end">
                <button
                  type="button"
                  onClick={() =>
                    setLab((prev) => ({
                      ...prev,
                      flashAttention: !prev.flashAttention,
                    }))
                  }
                  className={`button w-full rounded-xl px-2 py-2 font-mono text-[11px] ${
                    lab.flashAttention
                      ? "button--primary px-2 py-2"
                      : "button--neutral px-2 py-2"
                  }`}
                >
                  FA-2 {lab.flashAttention ? "ON" : "OFF"}
                </button>
              </div>
            </div>

            {/* Expandable Architecture Overrides (Layers, Hidden, KV Heads, Head Dim) */}
            <div className="border-t border-[var(--soft-border)] pt-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setShowArchOverrides((v) => !v)}
                  className="inline-flex items-center gap-1.5 font-mono text-[11px] font-bold text-[var(--soft-blue)]"
                >
                  {showArchOverrides ? (
                    <ChevronUp className="h-3.5 w-3.5" />
                  ) : (
                    <ChevronDown className="h-3.5 w-3.5" />
                  )}
                  <span>
                    {result.isAutoArch ? "Auto Architecture" : "Custom Overrides"}:{" "}
                    {result.architecture.layers}L · d={result.architecture.hiddenSize} ·{" "}
                    {result.architecture.kvHeads} KV · {result.architecture.headDim}d
                  </span>
                </button>

                {!result.isAutoArch && (
                  <button
                    type="button"
                    onClick={() =>
                      setLab((prev) => ({
                        ...prev,
                        overrideLayers: undefined,
                        overrideHiddenSize: undefined,
                        overrideKvHeads: undefined,
                        overrideHeadDim: undefined,
                      }))
                    }
                    className="inline-flex items-center gap-1 font-mono text-[10px] font-bold text-[var(--soft-orange-text)]"
                  >
                    <RotateCcw className="h-3 w-3" />
                    Reset Auto
                  </button>
                )}
              </div>

              {showArchOverrides && (
                <div className="mt-2.5 grid grid-cols-2 gap-2.5 sm:grid-cols-4">
                  <div>
                    <label
                      htmlFor="lab-ov-layers"
                      className="muted block font-mono text-[10px]"
                    >
                      Layers
                    </label>
                    <input
                      id="lab-ov-layers"
                      type="number"
                      min={1}
                      max={1024}
                      value={lab.overrideLayers ?? result.architecture.layers}
                      onChange={(e) => {
                        const v = parseInt(e.target.value, 10);
                        setLab((prev) => ({
                          ...prev,
                          overrideLayers: Number.isFinite(v) ? v : undefined,
                        }));
                      }}
                      className="surface mt-0.5 w-full rounded-xl px-2.5 py-1.5 text-right font-mono text-xs font-bold tabular"
                    />
                  </div>
                  <div>
                    <label
                      htmlFor="lab-ov-hidden"
                      className="muted block font-mono text-[10px]"
                    >
                      Hidden Dim
                    </label>
                    <input
                      id="lab-ov-hidden"
                      type="number"
                      min={128}
                      max={65536}
                      step={128}
                      value={lab.overrideHiddenSize ?? result.architecture.hiddenSize}
                      onChange={(e) => {
                        const v = parseInt(e.target.value, 10);
                        setLab((prev) => ({
                          ...prev,
                          overrideHiddenSize: Number.isFinite(v) ? v : undefined,
                        }));
                      }}
                      className="surface mt-0.5 w-full rounded-xl px-2.5 py-1.5 text-right font-mono text-xs font-bold tabular"
                    />
                  </div>
                  <div>
                    <label
                      htmlFor="lab-ov-kvh"
                      className="muted block font-mono text-[10px]"
                    >
                      KV Heads
                    </label>
                    <input
                      id="lab-ov-kvh"
                      type="number"
                      min={1}
                      max={512}
                      value={lab.overrideKvHeads ?? result.architecture.kvHeads}
                      onChange={(e) => {
                        const v = parseInt(e.target.value, 10);
                        setLab((prev) => ({
                          ...prev,
                          overrideKvHeads: Number.isFinite(v) ? v : undefined,
                        }));
                      }}
                      className="surface mt-0.5 w-full rounded-xl px-2.5 py-1.5 text-right font-mono text-xs font-bold tabular"
                    />
                  </div>
                  <div>
                    <label
                      htmlFor="lab-ov-hdim"
                      className="muted block font-mono text-[10px]"
                    >
                      Head Dim
                    </label>
                    <input
                      id="lab-ov-hdim"
                      type="number"
                      min={32}
                      max={512}
                      step={32}
                      value={lab.overrideHeadDim ?? result.architecture.headDim}
                      onChange={(e) => {
                        const v = parseInt(e.target.value, 10);
                        setLab((prev) => ({
                          ...prev,
                          overrideHeadDim: Number.isFinite(v) ? v : undefined,
                        }));
                      }}
                      className="surface mt-0.5 w-full rounded-xl px-2.5 py-1.5 text-right font-mono text-xs font-bold tabular"
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Well 3: Hardware Pool, Multi-GPU, Memory Ratio & Bandwidth */}
          <div className="well space-y-3.5 rounded-2xl p-4 lg:col-span-2">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="font-mono text-xs font-bold text-[var(--soft-text)]">
                  Hardware Memory Pool: {result.sanitized.gpuCount}× {result.sanitized.vramPerGpuGB} GB ={" "}
                  {result.totalRawPoolGB.toFixed(0)} GB Raw ({result.usablePoolGB.toFixed(1)} GB Usable @{" "}
                  {Math.round(result.sanitized.usableMemoryRatio * 100)}%)
                </p>
              </div>

              <div className="w-full sm:w-80">
                <select
                  value={selectedDevicePresetId}
                  onChange={(e) => applyHardwareDevice(e.target.value)}
                  aria-label="Fill Formula Lab specs from real GPU or Apple Silicon hardware"
                  className="surface w-full rounded-xl px-3 py-1.5 font-mono text-[11px] font-semibold"
                >
                  <option value="">Load real GPU / Apple Silicon specs...</option>
                  {vendorGroups.map((group) => (
                    <optgroup key={group.label} label={group.label}>
                      {group.devices.map((dev) => (
                        <option key={dev.id} value={dev.id}>
                          {dev.name} ({dev.vramGB} GB · {dev.bandwidthGBs} GB/s)
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div>
                <label
                  htmlFor="lab-vram-per-gpu"
                  className="muted mb-1 block font-mono text-[10px] font-bold uppercase"
                >
                  VRAM / GPU (GB)
                </label>
                <input
                  id="lab-vram-per-gpu"
                  type="number"
                  min={1}
                  max={16384}
                  step="any"
                  value={lab.vramPerGpuGB}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (Number.isFinite(v)) {
                      setSelectedDevicePresetId("");
                      setLab((prev) => ({ ...prev, vramPerGpuGB: v }));
                    }
                  }}
                  className="surface w-full rounded-xl px-3 py-2 text-right font-mono text-xs font-bold tabular"
                />
              </div>

              <div>
                <label
                  htmlFor="lab-gpu-count"
                  className="muted mb-1 block font-mono text-[10px] font-bold uppercase"
                >
                  GPU Count (×)
                </label>
                <input
                  id="lab-gpu-count"
                  type="number"
                  min={1}
                  max={256}
                  step={1}
                  value={lab.gpuCount}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (Number.isFinite(v)) {
                      setLab((prev) => ({ ...prev, gpuCount: v }));
                    }
                  }}
                  className="surface w-full rounded-xl px-3 py-2 text-right font-mono text-xs font-bold tabular"
                />
              </div>

              <div>
                <label
                  htmlFor="lab-bandwidth"
                  className="muted mb-1 block font-mono text-[10px] font-bold uppercase"
                >
                  Bandwidth (GB/s)
                </label>
                <input
                  id="lab-bandwidth"
                  type="number"
                  min={10}
                  max={50000}
                  step={10}
                  value={lab.bandwidthGBs}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (Number.isFinite(v)) {
                      setLab((prev) => ({ ...prev, bandwidthGBs: v }));
                    }
                  }}
                  className="surface w-full rounded-xl px-3 py-2 text-right font-mono text-xs font-bold tabular"
                />
              </div>

              <div>
                <label
                  htmlFor="lab-usable-ratio"
                  className="muted mb-1 block font-mono text-[10px] font-bold uppercase"
                >
                  Usable Ratio (%)
                </label>
                <input
                  id="lab-usable-ratio"
                  type="number"
                  min={40}
                  max={99}
                  step={1}
                  value={Math.round(lab.usableMemoryRatio * 100)}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (Number.isFinite(v)) {
                      setLab((prev) => ({ ...prev, usableMemoryRatio: v / 100 }));
                    }
                  }}
                  className="surface w-full rounded-xl px-3 py-2 text-right font-mono text-xs font-bold tabular"
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap gap-1.5">
                {QUICK_VRAM_CHIPS.map((v) => (
                  <button
                    key={v}
                    type="button"
                    onClick={() => {
                      setSelectedDevicePresetId("");
                      setLab((prev) => ({ ...prev, vramPerGpuGB: v }));
                    }}
                    className={`button rounded-lg px-2 py-1 font-mono text-[10px] ${
                      lab.vramPerGpuGB === v
                        ? "button--primary px-2 py-1"
                        : "button--neutral px-2 py-1"
                    }`}
                  >
                    {v}G
                  </button>
                ))}
              </div>

              <div className="flex flex-wrap gap-1.5">
                {QUICK_GPU_COUNTS.map((cnt) => (
                  <button
                    key={cnt}
                    type="button"
                    onClick={() => setLab((prev) => ({ ...prev, gpuCount: cnt }))}
                    className={`button rounded-lg px-2 py-1 font-mono text-[10px] ${
                      lab.gpuCount === cnt
                        ? "button--primary px-2 py-1"
                        : "button--neutral px-2 py-1"
                    }`}
                  >
                    {cnt}×
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Live KPIs + Stacked Proportional Bar + Equations */}
        <div className="mt-5 space-y-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="well rounded-xl p-3.5">
              <span className="muted block font-mono text-[10px] font-bold uppercase">
                Total VRAM Needed
              </span>
              <span
                className={`mt-1 block font-mono text-lg font-bold tabular ${
                  result.fits ? "stat--up" : "stat--warn"
                }`}
              >
                {result.totalRequiredGB.toFixed(2)} GB
              </span>
              <span className="muted block font-mono text-[10px] tabular">
                {result.fits
                  ? `+${result.headroomGB.toFixed(1)} GB headroom`
                  : `${Math.abs(result.headroomGB).toFixed(1)} GB over budget`}
              </span>
            </div>

            <div className="well rounded-xl p-3.5">
              <span className="muted block font-mono text-[10px] font-bold uppercase">
                Max Context Fit
              </span>
              <span className="mt-1 block font-mono text-lg font-bold tabular text-[var(--soft-blue)]">
                {result.maxContextThatFits > 0
                  ? result.maxContextThatFits.toLocaleString()
                  : "0 (OOM Weights)"}
              </span>
              <span className="muted block font-mono text-[10px] tabular">
                {result.kvPer1kTokensMB.toFixed(1)} MB per 1K tokens
              </span>
            </div>

            <div className="well rounded-xl p-3.5">
              <span className="muted block font-mono text-[10px] font-bold uppercase">
                Est. Decode / Prefill
              </span>
              <span className="mt-1 block font-mono text-lg font-bold tabular text-[var(--soft-text)]">
                {result.estimatedDecodeTps} tok/s
              </span>
              <span className="muted block font-mono text-[10px] tabular">
                Prefill ~{result.estimatedPrefillTps.toLocaleString()} tok/s
              </span>
            </div>

            <div className="well rounded-xl p-3.5">
              <span className="muted block font-mono text-[10px] font-bold uppercase">
                Min GPUs Needed
              </span>
              <span className="mt-1 block font-mono text-lg font-bold tabular text-[var(--soft-text)]">
                {result.minGpusNeeded}× {result.sanitized.vramPerGpuGB} GB
              </span>
              <span className="muted block font-mono text-[10px] tabular">
                Current cluster: {result.sanitized.gpuCount}×
              </span>
            </div>
          </div>

          {/* Stacked Memory Bar + Live Equations */}
          <div className="well space-y-3 rounded-2xl p-4">
            <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] font-semibold">
              <span>Memory Split (Weights vs KV Cache vs Overhead)</span>
              <div className="flex flex-wrap gap-3 tabular">
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-[#2563EB]" />
                  Weights: {result.weightsGB.toFixed(2)} GB ({result.weightsSharePct.toFixed(1)}%)
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-[#EA580C]" />
                  KV Cache: {result.kvCacheGB.toFixed(2)} GB ({result.kvSharePct.toFixed(1)}%)
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-[#22C55E]" />
                  Overhead: {result.overheadGB.toFixed(2)} GB ({result.overheadSharePct.toFixed(1)}%)
                </span>
              </div>
            </div>

            <div className="flex h-3 w-full overflow-hidden rounded-full border border-[var(--soft-border)] bg-[var(--soft-bg)]">
              <div
                style={{ width: `${Math.max(1, result.weightsSharePct)}%`, backgroundColor: "#2563EB" }}
                title={`Weights: ${result.weightsGB.toFixed(2)} GB`}
              />
              <div
                style={{ width: `${Math.max(0.5, result.kvSharePct)}%`, backgroundColor: "#EA580C" }}
                title={`KV Cache: ${result.kvCacheGB.toFixed(2)} GB`}
              />
              <div
                style={{ width: `${Math.max(0.5, result.overheadSharePct)}%`, backgroundColor: "#22C55E" }}
                title={`Overhead: ${result.overheadGB.toFixed(2)} GB`}
              />
            </div>

            <div className="grid grid-cols-1 gap-2 border-t border-[var(--soft-border)] pt-3 md:grid-cols-2">
              {result.liveEquations.map((eq) => (
                <div
                  key={eq.label}
                  className="surface flex flex-col justify-between gap-1 rounded-xl px-3 py-2 font-mono text-[11px]"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-bold text-[var(--soft-blue)]">{eq.label}</span>
                    <span className="font-bold tabular text-[var(--soft-text)]">
                      = {eq.result}
                    </span>
                  </div>
                  <span className="muted truncate text-[10px]">{eq.expression}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Interactive Scaling Sweep Matrix (0.5B->2T or 4K->2M Context) */}
          <div className="well space-y-3 rounded-2xl p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="text-xs font-bold text-[var(--soft-text)]">
                  Interactive Scaling Sweep Matrix (Click any row to apply)
                </h2>
                <p className="muted text-[11px]">
                  Compare footprint across tiny to multi-trillion parameter counts or long-context regimes.
                </p>
              </div>
              <div className="flex gap-1.5">
                <button
                  type="button"
                  onClick={() => setSweepMode("params")}
                  className={`button rounded-lg px-3 py-1.5 font-mono text-[11px] ${
                    sweepMode === "params" ? "button--primary px-3 py-1.5" : "button--neutral px-3 py-1.5"
                  }`}
                >
                  0.5B → 2T Params
                </button>
                <button
                  type="button"
                  onClick={() => setSweepMode("context")}
                  className={`button rounded-lg px-3 py-1.5 font-mono text-[11px] ${
                    sweepMode === "context" ? "button--primary px-3 py-1.5" : "button--neutral px-3 py-1.5"
                  }`}
                >
                  4K → 2M Context
                </button>
              </div>
            </div>

            <div className="soft-scroll overflow-x-auto">
              <table className="w-full text-left font-mono text-[11px] tabular">
                <thead>
                  <tr className="muted border-b border-[var(--soft-border)] text-[10px] uppercase">
                    <th className="py-2 pr-3">
                      {sweepMode === "params" ? "Model Scale" : "Context Length"}
                    </th>
                    <th className="py-2 pr-3">Weights</th>
                    <th className="py-2 pr-3">KV Cache</th>
                    <th className="py-2 pr-3">Total Needed</th>
                    <th className="py-2 pr-3">Est. Decode</th>
                    <th className="py-2">Fit / Min GPUs</th>
                  </tr>
                </thead>
                <tbody>
                  {sweepMode === "params"
                    ? paramSweep.map((row) => {
                        const active = Math.abs(lab.totalParamsB - row.paramsB) < 0.01;
                        return (
                          <tr
                            key={row.paramsB}
                            onClick={() => updateParamsB(row.paramsB)}
                            className={`cursor-pointer border-b border-[var(--soft-border)]/60 transition hover:bg-[var(--soft-shadow)]/30 ${
                              active ? "bg-[var(--soft-blue-fill)]/12 font-bold" : ""
                            }`}
                          >
                            <td className="py-2 pr-3 font-bold text-[var(--soft-blue)]">
                              {row.label}
                            </td>
                            <td className="py-2 pr-3">{row.weightsGB.toFixed(1)} GB</td>
                            <td className="py-2 pr-3">{row.kvCacheGB.toFixed(1)} GB</td>
                            <td className="py-2 pr-3 font-semibold">
                              {row.totalRequiredGB.toFixed(1)} GB
                            </td>
                            <td className="py-2 pr-3">{row.decodeTps} t/s</td>
                            <td
                              className={`py-2 font-bold ${
                                row.fits ? "stat--up" : "stat--warn"
                              }`}
                            >
                              {row.fits ? "FITS" : `OOM (min ${row.minGpusNeeded}×)`}
                            </td>
                          </tr>
                        );
                      })
                    : contextSweep.map((row) => {
                        const active = lab.contextLength === row.contextLength;
                        return (
                          <tr
                            key={row.contextLength}
                            onClick={() =>
                              setLab((prev) => ({
                                ...prev,
                                contextLength: row.contextLength,
                              }))
                            }
                            className={`cursor-pointer border-b border-[var(--soft-border)]/60 transition hover:bg-[var(--soft-shadow)]/30 ${
                              active ? "bg-[var(--soft-blue-fill)]/12 font-bold" : ""
                            }`}
                          >
                            <td className="py-2 pr-3 font-bold text-[var(--soft-blue)]">
                              {row.label} ({row.contextLength.toLocaleString()})
                            </td>
                            <td className="py-2 pr-3">{row.weightsGB.toFixed(1)} GB</td>
                            <td className="py-2 pr-3">{row.kvCacheGB.toFixed(1)} GB</td>
                            <td className="py-2 pr-3 font-semibold">
                              {row.totalRequiredGB.toFixed(1)} GB
                            </td>
                            <td className="py-2 pr-3">{row.decodeTps} t/s</td>
                            <td
                              className={`py-2 font-bold ${
                                row.fits ? "stat--up" : "stat--warn"
                              }`}
                            >
                              {row.fits ? "FITS" : `OOM (min ${row.minGpusNeeded}×)`}
                            </td>
                          </tr>
                        );
                      })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* ENGINEERING REFERENCE DIRECTORY + SEARCH + CATEGORY FILTER */}
      <section className="surface p-6">
        <div className="flex flex-col justify-between gap-4 border-b border-[var(--soft-border)] pb-4 lg:flex-row lg:items-center">
          <div className="flex items-start gap-3">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-[var(--soft-blue-fill)] text-white">
              <BookOpen className="h-5 w-5" />
            </div>
            <div>
              <span className="font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[var(--soft-blue)]">
                LocalAI Hardware Engineering Reference
              </span>
              <h2 className="text-lg font-bold text-[var(--soft-text)]">
                Understanding CPU, GPU, RAM &amp; Context Window Parameters for AI
              </h2>
              <p className="muted mt-0.5 text-xs">
                Every hardware metric mapped to real inference bottlenecks across decoding, prefill, and multi-GPU scaling.
              </p>
            </div>
          </div>

          <div className="relative w-full lg:w-72">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--soft-muted)]" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search bandwidth, KV cache, MLA, PCIe..."
              className="well w-full rounded-xl py-2.5 pl-9 pr-3 text-xs text-[var(--soft-text)] placeholder:text-[var(--soft-muted)]"
              aria-label="Search AI hardware help parameters"
            />
          </div>
        </div>

        <div className="mt-5 flex flex-wrap gap-2">
          {CATEGORIES.map((c) => {
            const active = category === c.id;
            return (
              <button
                key={c.id}
                type="button"
                onClick={() => setCategory(c.id)}
                className={`button px-3.5 py-2 text-xs ${
                  active ? "button--primary" : "button--neutral"
                }`}
              >
                {c.label}
              </button>
            );
          })}
        </div>
      </section>

      {/* Parameter Cards Grid */}
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
        {filteredEntries.map((item) => {
          const Icon =
            item.category === "gpu"
              ? Zap
              : item.category === "ram"
                ? HardDrive
                : item.category === "cpu"
                  ? Cpu
                  : Sparkles;

          const badgeBg =
            item.category === "gpu"
              ? "#2563EB"
              : item.category === "ram"
                ? "#15803D"
                : item.category === "cpu"
                  ? "#C2410C"
                  : "#1F2937";

          return (
            <article
              key={item.id}
              className="surface flex flex-col justify-between p-5"
            >
              <div>
                <div className="flex items-center justify-between gap-2">
                  <span
                    className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 font-mono text-[10px] font-bold uppercase text-white"
                    style={{ backgroundColor: badgeBg }}
                  >
                    <Icon className="h-3 w-3" />
                    <span>{item.categoryLabel}</span>
                  </span>
                  <span className="rounded-lg border border-[var(--soft-border)] px-2.5 py-0.5 font-mono text-[10px] font-semibold text-[var(--soft-muted)]">
                    {item.impact}
                  </span>
                </div>

                <h3 className="mt-3 text-base font-bold text-[var(--soft-text)]">
                  {item.title}
                </h3>
                <p className="muted mt-1.5 text-xs leading-relaxed">
                  {item.summary}
                </p>

                <div className="mt-3 border-t border-[var(--soft-border)] pt-3">
                  <div className="font-mono text-[10px] font-bold uppercase tracking-wider text-[var(--soft-blue)]">
                    Why It Matters for LocalAI
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-[var(--soft-text)]">
                    {item.whyAI}
                  </p>
                </div>
              </div>

              <div className="well mt-4 rounded-xl p-3 font-mono text-[11px] font-semibold text-[var(--soft-text)]">
                <Layers className="mr-1.5 inline h-3.5 w-3.5 text-[var(--soft-blue)]" />
                {item.formulaOrRule}
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
