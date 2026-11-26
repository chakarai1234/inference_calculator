export type LabAttentionMode = "mha" | "gqa-4" | "gqa-8" | "mqa" | "mla";

export type LabKVPrecision = "FP16" | "FP8" | "Q8_0" | "Q4_0";

export interface LabQuantPreset {
  id: string;
  label: string;
  bpw: number;
  note: string;
}

export interface FormulaLabInput {
  totalParamsB: number;
  isMoE: boolean;
  activeParamsB: number;
  quantLabel: string;
  bitsPerWeight: number;
  contextLength: number;
  batchSize: number;
  kvPrecision: LabKVPrecision;
  attentionMode: LabAttentionMode;
  flashAttention: boolean;
  vramPerGpuGB: number;
  gpuCount: number;
  usableMemoryRatio: number;
  bandwidthGBs: number;
  fp16TflopsPerGpu: number;
  overrideLayers?: number;
  overrideHiddenSize?: number;
  overrideKvHeads?: number;
  overrideHeadDim?: number;
}

export interface LabScalePreset {
  id: string;
  label: string;
  badge: string;
  input: Partial<FormulaLabInput>;
}

export interface LabAutoArchitecture {
  layers: number;
  hiddenSize: number;
  qHeads: number;
  kvHeads: number;
  headDim: number;
  attentionLabel: string;
}

export interface LabEquationLine {
  label: string;
  expression: string;
  result: string;
}

export interface FormulaLabResult {
  sanitized: FormulaLabInput;
  architecture: LabAutoArchitecture;
  isAutoArch: boolean;
  weightsGB: number;
  kvCacheGB: number;
  kvPer1kTokensMB: number;
  kvBytesPerToken: number;
  overheadGB: number;
  totalRequiredGB: number;
  totalRawPoolGB: number;
  usablePoolGB: number;
  headroomGB: number;
  fits: boolean;
  utilizationPct: number;
  minGpusNeeded: number;
  maxContextThatFits: number;
  estimatedDecodeTps: number;
  estimatedPrefillTps: number;
  weightsSharePct: number;
  kvSharePct: number;
  overheadSharePct: number;
  liveEquations: LabEquationLine[];
}

export interface ParameterSweepRow {
  paramsB: number;
  label: string;
  weightsGB: number;
  kvCacheGB: number;
  totalRequiredGB: number;
  decodeTps: number;
  fits: boolean;
  minGpusNeeded: number;
}

export interface ContextSweepRow {
  contextLength: number;
  label: string;
  weightsGB: number;
  kvCacheGB: number;
  totalRequiredGB: number;
  decodeTps: number;
  fits: boolean;
  minGpusNeeded: number;
}

const BYTES_PER_GIB = 1024 * 1024 * 1024;
const BYTES_PER_MIB = 1024 * 1024;
const MLA_LATENT_ELEMENTS_PER_LAYER = 576; // kv_lora_rank=512 + qk_rope_head_dim=64

export const LAB_KV_BYTES: Record<LabKVPrecision, number> = {
  FP16: 2.0,
  FP8: 1.0,
  Q8_0: 1.06,
  Q4_0: 0.56,
};

export const LAB_ATTENTION_LABELS: Record<LabAttentionMode, string> = {
  mha: "MHA (1:1 Q/KV)",
  "gqa-4": "GQA 4:1",
  "gqa-8": "GQA 8:1",
  mqa: "MQA (1 KV Head)",
  mla: "DeepSeek MLA (576d Latent)",
};

export const LAB_QUANT_PRESETS: LabQuantPreset[] = [
  { id: "FP32", label: "FP32", bpw: 32.0, note: "Full 32-bit float reference" },
  { id: "FP16", label: "FP16 / BF16", bpw: 16.0, note: "Uncompressed baseline weights" },
  { id: "Q8_0", label: "Q8_0", bpw: 8.5, note: "Near-lossless 8-bit GGUF" },
  { id: "FP8", label: "FP8 (E4M3)", bpw: 8.2, note: "Hopper / Ada / Blackwell FP8" },
  { id: "Q6_K", label: "Q6_K", bpw: 6.56, note: "High-fidelity 6-bit K-quant" },
  { id: "Q5_K_M", label: "Q5_K_M", bpw: 5.68, note: "Balanced 5-bit K-quant" },
  { id: "Q4_K_M", label: "Q4_K_M", bpw: 4.85, note: "Standard local inference sweet spot" },
  { id: "IQ4_XS", label: "IQ4_XS", bpw: 4.25, note: "Importance-weighted 4-bit" },
  { id: "Q3_K_M", label: "Q3_K_M", bpw: 3.91, note: "Tight VRAM 3-bit K-quant" },
  { id: "Q2_K", label: "Q2_K", bpw: 3.35, note: "Extreme 2-bit K-quant" },
  { id: "IQ2_XXS", label: "IQ2_XXS", bpw: 2.1, note: "Ultra-compressed 2.1 bpw" },
  { id: "BITNET", label: "BitNet 1.58b", bpw: 1.58, note: "Ternary {-1, 0, +1} weights" },
];

export const DEFAULT_FORMULA_LAB_INPUT: FormulaLabInput = {
  totalParamsB: 8,
  isMoE: false,
  activeParamsB: 8,
  quantLabel: "Q4_K_M",
  bitsPerWeight: 4.85,
  contextLength: 8192,
  batchSize: 1,
  kvPrecision: "FP16",
  attentionMode: "gqa-8",
  flashAttention: true,
  vramPerGpuGB: 16,
  gpuCount: 1,
  usableMemoryRatio: 0.94,
  bandwidthGBs: 448,
  fp16TflopsPerGpu: 80,
};

export const LAB_SCALE_PRESETS: LabScalePreset[] = [
  {
    id: "0.5b-edge",
    label: "0.5B Edge",
    badge: "Sub-1B",
    input: {
      totalParamsB: 0.5,
      isMoE: false,
      activeParamsB: 0.5,
      quantLabel: "Q4_K_M",
      bitsPerWeight: 4.85,
      contextLength: 32768,
      attentionMode: "gqa-4",
      kvPrecision: "FP16",
      vramPerGpuGB: 8,
      gpuCount: 1,
      usableMemoryRatio: 0.92,
      bandwidthGBs: 256,
      fp16TflopsPerGpu: 22,
    },
  },
  {
    id: "3b-mobile",
    label: "3B Mobile",
    badge: "Dense",
    input: {
      totalParamsB: 3,
      isMoE: false,
      activeParamsB: 3,
      quantLabel: "Q4_K_M",
      bitsPerWeight: 4.85,
      contextLength: 32768,
      attentionMode: "gqa-4",
      kvPrecision: "FP16",
      vramPerGpuGB: 12,
      gpuCount: 1,
      usableMemoryRatio: 0.94,
      bandwidthGBs: 360,
      fp16TflopsPerGpu: 45,
    },
  },
  {
    id: "8b-desktop",
    label: "8B Desktop",
    badge: "Dense",
    input: {
      totalParamsB: 8,
      isMoE: false,
      activeParamsB: 8,
      quantLabel: "Q4_K_M",
      bitsPerWeight: 4.85,
      contextLength: 8192,
      attentionMode: "gqa-8",
      kvPrecision: "FP16",
      vramPerGpuGB: 16,
      gpuCount: 1,
      usableMemoryRatio: 0.94,
      bandwidthGBs: 448,
      fp16TflopsPerGpu: 80,
    },
  },
  {
    id: "32b-workstation",
    label: "32B Workstation",
    badge: "Dense",
    input: {
      totalParamsB: 32,
      isMoE: false,
      activeParamsB: 32,
      quantLabel: "Q4_K_M",
      bitsPerWeight: 4.85,
      contextLength: 32768,
      attentionMode: "gqa-8",
      kvPrecision: "FP16",
      vramPerGpuGB: 24,
      gpuCount: 1,
      usableMemoryRatio: 0.94,
      bandwidthGBs: 1008,
      fp16TflopsPerGpu: 165,
    },
  },
  {
    id: "70b-server",
    label: "70B Server",
    badge: "Dense",
    input: {
      totalParamsB: 70,
      isMoE: false,
      activeParamsB: 70,
      quantLabel: "Q4_K_M",
      bitsPerWeight: 4.85,
      contextLength: 32768,
      attentionMode: "gqa-8",
      kvPrecision: "FP16",
      vramPerGpuGB: 64,
      gpuCount: 1,
      usableMemoryRatio: 0.75,
      bandwidthGBs: 546,
      fp16TflopsPerGpu: 112,
    },
  },
  {
    id: "235b-moe",
    label: "235B MoE (22B act)",
    badge: "MoE",
    input: {
      totalParamsB: 235,
      isMoE: true,
      activeParamsB: 22,
      quantLabel: "Q4_K_M",
      bitsPerWeight: 4.85,
      contextLength: 65536,
      attentionMode: "gqa-8",
      kvPrecision: "FP16",
      vramPerGpuGB: 192,
      gpuCount: 1,
      usableMemoryRatio: 0.85,
      bandwidthGBs: 800,
      fp16TflopsPerGpu: 135,
    },
  },
  {
    id: "405b-frontier",
    label: "405B Frontier",
    badge: "8x H100",
    input: {
      totalParamsB: 405,
      isMoE: false,
      activeParamsB: 405,
      quantLabel: "FP8 (E4M3)",
      bitsPerWeight: 8.2,
      contextLength: 131072,
      attentionMode: "gqa-8",
      kvPrecision: "FP8",
      vramPerGpuGB: 80,
      gpuCount: 8,
      usableMemoryRatio: 0.95,
      bandwidthGBs: 3350,
      fp16TflopsPerGpu: 989,
    },
  },
  {
    id: "671b-deepseek-mla",
    label: "671B DeepSeek MLA",
    badge: "MLA MoE",
    input: {
      totalParamsB: 671,
      isMoE: true,
      activeParamsB: 37,
      quantLabel: "Q4_K_M",
      bitsPerWeight: 4.85,
      contextLength: 131072,
      attentionMode: "mla",
      kvPrecision: "FP8",
      vramPerGpuGB: 512,
      gpuCount: 1,
      usableMemoryRatio: 0.85,
      bandwidthGBs: 819,
      fp16TflopsPerGpu: 135,
    },
  },
  {
    id: "1000b-1t-moe",
    label: "1000B (1T) MoE",
    badge: "1 Trillion",
    input: {
      totalParamsB: 1000,
      isMoE: true,
      activeParamsB: 64,
      quantLabel: "FP8 (E4M3)",
      bitsPerWeight: 8.2,
      contextLength: 131072,
      attentionMode: "mla",
      kvPrecision: "FP8",
      vramPerGpuGB: 192,
      gpuCount: 8,
      usableMemoryRatio: 0.95,
      bandwidthGBs: 8000,
      fp16TflopsPerGpu: 2250,
    },
  },
  {
    id: "2000b-2t-mega",
    label: "2000B (2T) Mega",
    badge: "2 Trillion",
    input: {
      totalParamsB: 2000,
      isMoE: true,
      activeParamsB: 160,
      quantLabel: "Q4_K_M",
      bitsPerWeight: 4.85,
      contextLength: 262144,
      attentionMode: "gqa-8",
      kvPrecision: "FP8",
      vramPerGpuGB: 141,
      gpuCount: 16,
      usableMemoryRatio: 0.95,
      bandwidthGBs: 4800,
      fp16TflopsPerGpu: 989,
    },
  },
];

interface ArchAnchor {
  paramsB: number;
  layers: number;
  hiddenSize: number;
  qHeads: number;
  headDim: number;
}

const ARCH_ANCHORS: ArchAnchor[] = [
  { paramsB: 0.1, layers: 12, hiddenSize: 768, qHeads: 12, headDim: 64 },
  { paramsB: 0.5, layers: 24, hiddenSize: 896, qHeads: 14, headDim: 64 },
  { paramsB: 1.5, layers: 28, hiddenSize: 1536, qHeads: 12, headDim: 128 },
  { paramsB: 3, layers: 36, hiddenSize: 2048, qHeads: 16, headDim: 128 },
  { paramsB: 8, layers: 32, hiddenSize: 4096, qHeads: 32, headDim: 128 },
  { paramsB: 14, layers: 48, hiddenSize: 5120, qHeads: 40, headDim: 128 },
  { paramsB: 32, layers: 64, hiddenSize: 5120, qHeads: 40, headDim: 128 },
  { paramsB: 70, layers: 80, hiddenSize: 8192, qHeads: 64, headDim: 128 },
  { paramsB: 120, layers: 96, hiddenSize: 10240, qHeads: 80, headDim: 128 },
  { paramsB: 235, layers: 94, hiddenSize: 4096, qHeads: 64, headDim: 128 },
  { paramsB: 405, layers: 126, hiddenSize: 16384, qHeads: 128, headDim: 128 },
  { paramsB: 671, layers: 61, hiddenSize: 7168, qHeads: 128, headDim: 128 },
  { paramsB: 1000, layers: 128, hiddenSize: 16384, qHeads: 128, headDim: 128 },
  { paramsB: 2000, layers: 160, hiddenSize: 20480, qHeads: 160, headDim: 128 },
  { paramsB: 5000, layers: 192, hiddenSize: 24576, qHeads: 192, headDim: 128 },
  { paramsB: 10000, layers: 224, hiddenSize: 28672, qHeads: 224, headDim: 128 },
];

function clampFinite(value: number, fallback: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

function roundToMultiple(value: number, multiple: number): number {
  return Math.max(multiple, Math.round(value / multiple) * multiple);
}

export function deriveAutoArchitecture(
  totalParamsB: number,
  attentionMode: LabAttentionMode
): LabAutoArchitecture {
  const safeParams = clampFinite(totalParamsB, 8, 0.01, 100000);

  let lower = ARCH_ANCHORS[0];
  let upper = ARCH_ANCHORS[ARCH_ANCHORS.length - 1];

  if (safeParams <= lower.paramsB) {
    upper = lower;
  } else if (safeParams >= upper.paramsB) {
    lower = upper;
  } else {
    for (let i = 0; i < ARCH_ANCHORS.length - 1; i++) {
      const a = ARCH_ANCHORS[i];
      const b = ARCH_ANCHORS[i + 1];
      if (safeParams >= a.paramsB && safeParams <= b.paramsB) {
        lower = a;
        upper = b;
        break;
      }
    }
  }

  let t = 0;
  if (upper.paramsB > lower.paramsB) {
    const logLow = Math.log10(lower.paramsB);
    const logHigh = Math.log10(upper.paramsB);
    t = (Math.log10(safeParams) - logLow) / (logHigh - logLow);
  }

  // Smooth extrapolation above top anchor
  const extraScale =
    safeParams > ARCH_ANCHORS[ARCH_ANCHORS.length - 1].paramsB
      ? Math.pow(safeParams / ARCH_ANCHORS[ARCH_ANCHORS.length - 1].paramsB, 0.25)
      : 1;

  const layers = Math.max(
    4,
    Math.round((lower.layers + (upper.layers - lower.layers) * t) * extraScale)
  );
  const hiddenSize = roundToMultiple(
    (lower.hiddenSize + (upper.hiddenSize - lower.hiddenSize) * t) * extraScale,
    128
  );
  const headDim = t < 0.5 ? lower.headDim : upper.headDim;
  const qHeads = Math.max(
    4,
    Math.round((lower.qHeads + (upper.qHeads - lower.qHeads) * t) * extraScale)
  );

  let kvHeads = qHeads;
  if (attentionMode === "gqa-4") {
    kvHeads = Math.max(1, Math.round(qHeads / 4));
  } else if (attentionMode === "gqa-8") {
    kvHeads = Math.max(1, Math.round(qHeads / 8));
  } else if (attentionMode === "mqa" || attentionMode === "mla") {
    kvHeads = 1;
  }

  return {
    layers,
    hiddenSize,
    qHeads,
    kvHeads,
    headDim,
    attentionLabel: LAB_ATTENTION_LABELS[attentionMode],
  };
}

export function sanitizeLabInput(input: FormulaLabInput): FormulaLabInput {
  const totalParamsB = clampFinite(input.totalParamsB, 8, 0.01, 100000);
  const isMoE = Boolean(input.isMoE);
  const activeParamsB = isMoE
    ? clampFinite(input.activeParamsB, Math.min(totalParamsB, 32), 0.01, totalParamsB)
    : totalParamsB;

  const bitsPerWeight = clampFinite(input.bitsPerWeight, 4.85, 0.5, 64);
  const contextLength = Math.round(clampFinite(input.contextLength, 8192, 128, 20_000_000));
  const batchSize = Math.round(clampFinite(input.batchSize, 1, 1, 512));
  const vramPerGpuGB = clampFinite(input.vramPerGpuGB, 16, 1, 16384);
  const gpuCount = Math.round(clampFinite(input.gpuCount, 1, 1, 256));
  const usableMemoryRatio = clampFinite(input.usableMemoryRatio, 0.94, 0.4, 0.99);
  const bandwidthGBs = clampFinite(input.bandwidthGBs, 448, 10, 50000);
  const fp16TflopsPerGpu = clampFinite(input.fp16TflopsPerGpu, 80, 1, 20000);

  return {
    ...input,
    totalParamsB,
    isMoE,
    activeParamsB,
    bitsPerWeight,
    contextLength,
    batchSize,
    vramPerGpuGB,
    gpuCount,
    usableMemoryRatio,
    bandwidthGBs,
    fp16TflopsPerGpu,
  };
}

export function formatParamScaleShort(paramsB: number): string {
  if (paramsB >= 1000) {
    const t = paramsB / 1000;
    return `${Number.isInteger(t) ? t : Number(t.toFixed(2))}T`;
  }
  if (paramsB < 1) {
    return `${Math.round(paramsB * 1000)}M`;
  }
  return `${Number.isInteger(paramsB) ? paramsB : Number(paramsB.toFixed(1))}B`;
}

export function formatContextShort(ctx: number): string {
  if (ctx >= 1_048_576 && ctx % 1_048_576 === 0) {
    return `${ctx / 1_048_576}M`;
  }
  if (ctx >= 1_000_000) {
    const m = ctx / 1_000_000;
    return `${Number.isInteger(m) ? m : Number(m.toFixed(2))}M`;
  }
  if (ctx >= 1024 && ctx % 1024 === 0) {
    return `${ctx / 1024}K`;
  }
  if (ctx >= 1000) {
    return `${(ctx / 1000).toFixed(1)}K`;
  }
  return String(ctx);
}

export function computeFormulaLab(rawInput: FormulaLabInput): FormulaLabResult {
  const input = sanitizeLabInput(rawInput);
  const autoArch = deriveAutoArchitecture(input.totalParamsB, input.attentionMode);

  const layers =
    input.overrideLayers !== undefined && Number.isFinite(input.overrideLayers)
      ? Math.round(clampFinite(input.overrideLayers, autoArch.layers, 1, 1024))
      : autoArch.layers;
  const hiddenSize =
    input.overrideHiddenSize !== undefined && Number.isFinite(input.overrideHiddenSize)
      ? Math.round(clampFinite(input.overrideHiddenSize, autoArch.hiddenSize, 128, 65536))
      : autoArch.hiddenSize;
  const kvHeads =
    input.overrideKvHeads !== undefined && Number.isFinite(input.overrideKvHeads)
      ? Math.round(clampFinite(input.overrideKvHeads, autoArch.kvHeads, 1, 512))
      : autoArch.kvHeads;
  const headDim =
    input.overrideHeadDim !== undefined && Number.isFinite(input.overrideHeadDim)
      ? Math.round(clampFinite(input.overrideHeadDim, autoArch.headDim, 32, 512))
      : autoArch.headDim;

  const isAutoArch =
    input.overrideLayers === undefined &&
    input.overrideHiddenSize === undefined &&
    input.overrideKvHeads === undefined &&
    input.overrideHeadDim === undefined;

  const architecture: LabAutoArchitecture = {
    layers,
    hiddenSize,
    qHeads: autoArch.qHeads,
    kvHeads,
    headDim,
    attentionLabel: autoArch.attentionLabel,
  };

  // 1. Model Weights GB
  const weightsGB =
    (input.totalParamsB * 1e9 * (input.bitsPerWeight / 8)) / BYTES_PER_GIB;

  // 2. KV Cache GB
  const kvBytesPerElement = LAB_KV_BYTES[input.kvPrecision] ?? 2.0;
  const kvBytesPerToken =
    input.attentionMode === "mla"
      ? layers * MLA_LATENT_ELEMENTS_PER_LAYER * kvBytesPerElement
      : 2 * layers * kvHeads * headDim * kvBytesPerElement;

  const kvCacheGB =
    (kvBytesPerToken * input.contextLength * input.batchSize) / BYTES_PER_GIB;
  const kvPer1kTokensMB =
    (kvBytesPerToken * 1024 * input.batchSize) / BYTES_PER_MIB;

  // 3. Activation & Multi-GPU Overhead GB
  const flashScale = input.flashAttention ? 0.62 : 1.0;
  const multiGpuOverhead = input.gpuCount > 1 ? 0.35 * (input.gpuCount - 1) : 0;
  const overheadGB =
    0.45 + (hiddenSize / 8192) * 0.45 * input.batchSize * flashScale + multiGpuOverhead;

  // 4. Total & Usable Pool
  const totalRequiredGB = weightsGB + kvCacheGB + overheadGB;
  const totalRawPoolGB = input.vramPerGpuGB * input.gpuCount;
  const usablePoolGB = totalRawPoolGB * input.usableMemoryRatio;
  const headroomGB = usablePoolGB - totalRequiredGB;
  const fits = headroomGB >= 0;
  const utilizationPct =
    usablePoolGB > 0 ? (totalRequiredGB / usablePoolGB) * 100 : 999;

  const usablePerGpu = Math.max(0.1, input.vramPerGpuGB * input.usableMemoryRatio);
  const minGpusNeeded = Math.max(1, Math.ceil(totalRequiredGB / usablePerGpu));

  // 5. Max Context That Fits
  const remainingForKvGB = usablePoolGB - weightsGB - overheadGB;
  const maxContextRaw =
    remainingForKvGB <= 0 || kvBytesPerToken <= 0
      ? 0
      : (remainingForKvGB * BYTES_PER_GIB) / (kvBytesPerToken * input.batchSize);
  const maxContextThatFits =
    maxContextRaw <= 0 ? 0 : Math.floor(maxContextRaw / 256) * 256;

  // 6. Throughput Estimation (Bandwidth-bound Decode & Compute-bound Prefill)
  const gpuScale = input.gpuCount === 1 ? 1 : Math.pow(input.gpuCount, 0.85);
  const activeWeightTrafficGB =
    (input.activeParamsB * 1e9 * (input.bitsPerWeight / 8)) / BYTES_PER_GIB;
  const decodeTrafficGB = Math.max(0.05, activeWeightTrafficGB + 0.18 * kvCacheGB);
  const estimatedDecodeTps = Number(
    Math.max(0.05, (input.bandwidthGBs * gpuScale * 0.76) / decodeTrafficGB).toFixed(1)
  );

  const prefillEfficiency = input.bitsPerWeight >= 16 ? 0.55 : 0.72;
  const estimatedPrefillTps = Number(
    Math.max(
      0.5,
      (input.fp16TflopsPerGpu * gpuScale * prefillEfficiency * 1000) /
        Math.max(0.5, 2 * input.activeParamsB)
    ).toFixed(0)
  );

  // Shares for stacked allocation bar
  const safeDenom = Math.max(0.0001, totalRequiredGB);
  const weightsSharePct = (weightsGB / safeDenom) * 100;
  const kvSharePct = (kvCacheGB / safeDenom) * 100;
  const overheadSharePct = (overheadGB / safeDenom) * 100;

  const kvFormulaDesc =
    input.attentionMode === "mla"
      ? `${layers}L × 576 (MLA) × ${kvBytesPerElement}B × ${input.contextLength.toLocaleString()} × B${input.batchSize}`
      : `2 × ${layers}L × ${kvHeads}KV × ${headDim}d × ${kvBytesPerElement}B × ${input.contextLength.toLocaleString()} × B${input.batchSize}`;

  const liveEquations: LabEquationLine[] = [
    {
      label: "Weights",
      expression: `(${input.totalParamsB}B × ${input.bitsPerWeight} bits) ÷ 8 ÷ 1.074`,
      result: `${weightsGB.toFixed(2)} GB`,
    },
    {
      label: "KV Cache",
      expression: kvFormulaDesc,
      result: `${kvCacheGB.toFixed(2)} GB (${kvPer1kTokensMB.toFixed(1)} MB / 1K ctx)`,
    },
    {
      label: "Activations + Overhead",
      expression: `0.45 + (${hiddenSize}/8192 × 0.45 × B${input.batchSize} × ${flashScale})${multiGpuOverhead > 0 ? ` + ${multiGpuOverhead.toFixed(2)}` : ""}`,
      result: `${overheadGB.toFixed(2)} GB`,
    },
    {
      label: "Decode Speed",
      expression: `(${input.bandwidthGBs} GB/s × ${gpuScale.toFixed(2)} × 0.76) ÷ (${activeWeightTrafficGB.toFixed(2)}G act + ${(0.18 * kvCacheGB).toFixed(2)}G kv)`,
      result: `${estimatedDecodeTps} tok/s`,
    },
  ];

  return {
    sanitized: input,
    architecture,
    isAutoArch,
    weightsGB,
    kvCacheGB,
    kvPer1kTokensMB,
    kvBytesPerToken,
    overheadGB,
    totalRequiredGB,
    totalRawPoolGB,
    usablePoolGB,
    headroomGB,
    fits,
    utilizationPct,
    minGpusNeeded,
    maxContextThatFits,
    estimatedDecodeTps,
    estimatedPrefillTps,
    weightsSharePct,
    kvSharePct,
    overheadSharePct,
    liveEquations,
  };
}

const SWEEP_PARAM_POINTS = [0.5, 3, 8, 14, 32, 70, 120, 235, 405, 671, 1000, 2000];
const SWEEP_CONTEXT_POINTS = [
  4096, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152,
];

export function buildParameterScaleSweep(input: FormulaLabInput): ParameterSweepRow[] {
  return SWEEP_PARAM_POINTS.map((paramsB) => {
    const activeRatio =
      input.isMoE && input.totalParamsB > 0
        ? Math.min(1, input.activeParamsB / input.totalParamsB)
        : 1;
    const res = computeFormulaLab({
      ...input,
      totalParamsB: paramsB,
      activeParamsB: input.isMoE ? Math.max(0.1, paramsB * activeRatio) : paramsB,
      overrideLayers: undefined,
      overrideHiddenSize: undefined,
      overrideKvHeads: undefined,
      overrideHeadDim: undefined,
    });
    return {
      paramsB,
      label: formatParamScaleShort(paramsB),
      weightsGB: res.weightsGB,
      kvCacheGB: res.kvCacheGB,
      totalRequiredGB: res.totalRequiredGB,
      decodeTps: res.estimatedDecodeTps,
      fits: res.fits,
      minGpusNeeded: res.minGpusNeeded,
    };
  });
}

export function buildContextScaleSweep(input: FormulaLabInput): ContextSweepRow[] {
  return SWEEP_CONTEXT_POINTS.map((ctx) => {
    const res = computeFormulaLab({
      ...input,
      contextLength: ctx,
    });
    return {
      contextLength: ctx,
      label: formatContextShort(ctx),
      weightsGB: res.weightsGB,
      kvCacheGB: res.kvCacheGB,
      totalRequiredGB: res.totalRequiredGB,
      decodeTps: res.estimatedDecodeTps,
      fits: res.fits,
      minGpusNeeded: res.minGpusNeeded,
    };
  });
}
