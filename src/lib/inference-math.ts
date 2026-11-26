import type {
  HardwareDevice,
  HFModelDetail,
  HFModelSummary,
  InferenceConfig,
  InferenceEstimate,
  KVCachePrecision,
  LocalAIBackend,
  ModelArchitectureConfig,
  QuantizationLevel,
  QuantizationSpec,
} from "@/types/inference";
import { getEffectiveMemoryGB } from "@/lib/hardware-catalog";

export const QUANTIZATION_SPECS: QuantizationSpec[] = [
  { id: "FP16", label: "FP16 / BF16", bpw: 16.0, qualityNote: "Uncompressed baseline weights" },
  { id: "FP8", label: "FP8 (E4M3)", bpw: 8.2, qualityNote: "Near-lossless on Ada/Hopper/Blackwell" },
  { id: "Q8_0", label: "Q8_0", bpw: 8.5, qualityNote: "Virtually identical to FP16 in llama.cpp" },
  { id: "Q6_K", label: "Q6_K", bpw: 6.56, qualityNote: "Very high fidelity 6-bit K-quant" },
  { id: "Q5_K_M", label: "Q5_K_M", bpw: 5.68, qualityNote: "Strong quality/memory balance" },
  { id: "Q4_K_M", label: "Q4_K_M", bpw: 4.85, qualityNote: "Standard LocalAI local sweet spot" },
  { id: "IQ4_XS", label: "IQ4_XS", bpw: 4.25, qualityNote: "Compact importance-matrix 4-bit" },
  { id: "Q3_K_M", label: "Q3_K_M", bpw: 3.91, qualityNote: "Tight VRAM fallback; slight perplexity rise" },
  { id: "Q2_K", label: "Q2_K", bpw: 3.35, qualityNote: "Extreme compression for oversized models" },
];

export const QUANT_MAP: Record<QuantizationLevel, QuantizationSpec> = Object.fromEntries(
  QUANTIZATION_SPECS.map((spec) => [spec.id, spec])
) as Record<QuantizationLevel, QuantizationSpec>;

export const KV_CACHE_BYTES: Record<KVCachePrecision, number> = {
  FP16: 2.0,
  Q8_0: 1.06,
  Q4_0: 0.56,
};

export const DEFAULT_INFERENCE_CONFIG: InferenceConfig = {
  quantization: "Q4_K_M",
  contextLength: 8192,
  batchSize: 1,
  kvCachePrecision: "FP16",
  flashAttention: true,
  gpuCount: 1,
  appleWiredRatio: 0.75,
};

const BYTES_PER_GIB = 1024 * 1024 * 1024;

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIdx = 0;
  while (value >= 1024 && unitIdx < units.length - 1) {
    value /= 1024;
    unitIdx += 1;
  }
  const decimals = unitIdx <= 1 ? 0 : value >= 100 ? 1 : 2;
  return `${value.toFixed(decimals)} ${units[unitIdx]}`;
}

export function formatParams(parametersB: number): string {
  if (!Number.isFinite(parametersB) || parametersB <= 0) return "Unknown";
  if (parametersB < 1) return `${Math.round(parametersB * 1000)}M`;
  if (parametersB >= 100) return `${Math.round(parametersB)}B`;
  return `${Number(parametersB.toFixed(1))}B`;
}

export function formatCompactNumber(num: number): string {
  if (!Number.isFinite(num) || num <= 0) return "0";
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`;
  return String(Math.round(num));
}

export function inferArchitectureFallback(model: HFModelSummary): ModelArchitectureConfig {
  if ("architecture" in model && (model as HFModelDetail).architecture) {
    return (model as HFModelDetail).architecture;
  }

  const p = Math.max(0.4, model.parametersB || 7);
  const embedded = model.embeddedConfig;
  const lowerId = `${model.id} ${model.architectureHint} ${model.tags.join(" ")}`.toLowerCase();

  const isDiffusion =
    model.pipelineTag === "text-to-image" ||
    model.formats.includes("diffusers") ||
    lowerId.includes("flux") ||
    lowerId.includes("stable-diffusion");

  if (isDiffusion) {
    return {
      archetype: "diffusion",
      source: "estimated",
      modelType: embedded?.modelType ?? "diffusion-transformer",
      architectureName: model.architectureHint || "Flux / Diffusion Pipeline",
      numLayers: embedded?.numLayers ?? 38,
      hiddenSize: embedded?.hiddenSize ?? 3072,
      intermediateSize: embedded?.intermediateSize ?? 12288,
      numAttentionHeads: embedded?.numAttentionHeads ?? 24,
      numKeyValueHeads: embedded?.numKeyValueHeads ?? 24,
      vocabSize: 32000,
      maxContextLength: 512,
      ropeTheta: 10000,
      tieWordEmbeddings: false,
    };
  }

  const hasVision =
    model.pipelineTag === "image-text-to-text" ||
    lowerId.includes("-vl") ||
    lowerId.includes("vision") ||
    lowerId.includes("gemma-3");

  // Estimate layer depth & hidden width from parameter scale when config.json isn't loaded yet
  let numLayers = 32;
  let hiddenSize = 4096;
  let numHeads = 32;
  let numKvHeads = 8;

  if (p <= 1.6) {
    numLayers = 22;
    hiddenSize = 2048;
    numHeads = 16;
    numKvHeads = 4;
  } else if (p <= 4) {
    numLayers = 28;
    hiddenSize = 2560;
    numHeads = 20;
    numKvHeads = 4;
  } else if (p <= 10) {
    numLayers = 32;
    hiddenSize = 4096;
    numHeads = 32;
    numKvHeads = 8;
  } else if (p <= 18) {
    numLayers = 44;
    hiddenSize = 5120;
    numHeads = 40;
    numKvHeads = 8;
  } else if (p <= 38) {
    numLayers = 64;
    hiddenSize = 5120;
    numHeads = 40;
    numKvHeads = 8;
  } else if (p <= 85) {
    numLayers = 80;
    hiddenSize = 8192;
    numHeads = 64;
    numKvHeads = 8;
  } else {
    numLayers = 64;
    hiddenSize = 7168;
    numHeads = 56;
    numKvHeads = 8;
  }

  const finalLayers = embedded?.numLayers ?? numLayers;
  const finalHidden = embedded?.hiddenSize ?? hiddenSize;
  const finalHeads = embedded?.numAttentionHeads ?? numHeads;
  const finalKvHeads = embedded?.numKeyValueHeads ?? numKvHeads;
  const finalInter = embedded?.intermediateSize ?? Math.round(finalHidden * 2.7);
  const isMoE = model.isMoE || (embedded?.numExperts ?? 0) > 1;

  return {
    archetype: isMoE ? "moe-transformer" : hasVision ? "multimodal-vlm" : "dense-transformer",
    source: embedded?.numLayers ? "config.json" : "estimated",
    modelType: embedded?.modelType ?? (model.architectureHint || "transformer"),
    architectureName: embedded?.architectureName ?? (model.architectureHint || "CausalLM"),
    numLayers: finalLayers,
    hiddenSize: finalHidden,
    intermediateSize: finalInter,
    numAttentionHeads: finalHeads,
    numKeyValueHeads: finalKvHeads,
    vocabSize: embedded?.vocabSize ?? 128256,
    maxContextLength: embedded?.maxContextLength ?? model.defaultContextLength ?? 32768,
    ropeTheta: embedded?.ropeTheta ?? 500000,
    tieWordEmbeddings: embedded?.tieWordEmbeddings ?? false,
    numExperts: embedded?.numExperts ?? (isMoE ? 8 : undefined),
    numActiveExperts: embedded?.numActiveExperts ?? (isMoE ? 2 : undefined),
    hasVisionEncoder: embedded?.hasVisionEncoder ?? hasVision,
    visionLayers: embedded?.visionLayers ?? (hasVision ? 24 : undefined),
    visionHiddenSize: embedded?.visionHiddenSize ?? (hasVision ? 1152 : undefined),
  };
}

function isDetailModel(model: HFModelSummary | HFModelDetail): model is HFModelDetail {
  return "quantVariants" in model && Array.isArray(model.quantVariants);
}

export function isModelFormatCompatible(
  model: HFModelSummary,
  device: HardwareDevice
): { supported: boolean; reason?: string; recommendedBackend: LocalAIBackend } {
  const isMlxOnly =
    model.formats.length === 1 && model.formats[0] === "mlx";

  if (isMlxOnly && device.vendor !== "apple") {
    return {
      supported: false,
      reason: "MLX format requires Apple Silicon",
      recommendedBackend: "mlx",
    };
  }

  const isDiffusion =
    model.pipelineTag === "text-to-image" || model.primaryFormat === "diffusers";

  if (isDiffusion) {
    if (!device.supportedBackends.includes("diffusers")) {
      return {
        supported: false,
        reason: `Diffusers backend unavailable on ${device.vendorLabel}`,
        recommendedBackend: "diffusers",
      };
    }
    return { supported: true, recommendedBackend: "diffusers" };
  }

  if (device.vendor === "apple" && model.formats.includes("mlx")) {
    return { supported: true, recommendedBackend: "mlx" };
  }

  if (model.formats.includes("gguf") && device.supportedBackends.includes("llama-cpp")) {
    return { supported: true, recommendedBackend: "llama-cpp" };
  }

  if (
    model.formats.includes("safetensors") &&
    device.supportedBackends.includes("vllm") &&
    device.vramGB >= 24 &&
    !device.unifiedMemory
  ) {
    return { supported: true, recommendedBackend: "vllm" };
  }

  if (device.supportedBackends.includes("llama-cpp")) {
    return { supported: true, recommendedBackend: "llama-cpp" };
  }

  if (device.supportedBackends.includes("transformers")) {
    return { supported: true, recommendedBackend: "transformers" };
  }

  return {
    supported: false,
    reason: "No compatible LocalAI backend on selected hardware",
    recommendedBackend: "llama-cpp",
  };
}

export function calculateInference(
  model: HFModelSummary | HFModelDetail,
  device: HardwareDevice,
  config: InferenceConfig
): InferenceEstimate {
  const arch = inferArchitectureFallback(model);
  const quantSpec = QUANT_MAP[config.quantization] ?? QUANT_MAP.Q4_K_M;
  const effectiveGpuCount = device.supportsMultiGpu
    ? Math.max(1, Math.min(config.gpuCount, device.maxGpuCount))
    : 1;

  const effectiveUsableMemoryGB = getEffectiveMemoryGB(
    device,
    effectiveGpuCount,
    config.appleWiredRatio
  );

  // 1. Exact matching weight file size if model detail has matching GGUF variant
  let weightsGB = (model.parametersB * 1e9 * (quantSpec.bpw / 8)) / BYTES_PER_GIB;
  let weightsSource: "exact-file" | "parameter-math" = "parameter-math";

  if (isDetailModel(model) && model.quantVariants.length > 0) {
    const exactMatch = model.quantVariants.find(
      (v) => v.mappedQuant === config.quantization
    );
    if (exactMatch && exactMatch.totalBytes > 0) {
      weightsGB = exactMatch.totalBytes / BYTES_PER_GIB;
      weightsSource = "exact-file";
    }
  } else if (
    config.quantization === "FP16" &&
    model.exactBaseBytes &&
    model.exactBaseBytes > 0
  ) {
    weightsGB = model.exactBaseBytes / BYTES_PER_GIB;
    weightsSource = "exact-file";
  }

  // 2. KV Cache size (GB)
  const effectiveContext =
    arch.archetype === "diffusion"
      ? 0
      : Math.min(config.contextLength, Math.max(arch.maxContextLength, 4096));

  const headDim = Math.max(
    64,
    Math.round(arch.hiddenSize / Math.max(1, arch.numAttentionHeads))
  );
  const kvBytesPerElement = KV_CACHE_BYTES[config.kvCachePrecision] ?? 2;
  const kvCacheBytesPerToken =
    2 * arch.numLayers * arch.numKeyValueHeads * headDim * kvBytesPerElement;

  const kvCacheGB =
    arch.archetype === "diffusion"
      ? 0
      : (kvCacheBytesPerToken * effectiveContext * config.batchSize) / BYTES_PER_GIB;

  // 3. Activation & runtime scratch buffer overhead (GB)
  const activationScale = config.flashAttention ? 0.62 : 1.0;
  const baseOverhead = arch.archetype === "diffusion" ? 1.85 : 0.48;
  const hiddenOverhead =
    (arch.hiddenSize / 8192) * 0.42 * config.batchSize * activationScale;
  const multiGpuOverhead = effectiveGpuCount > 1 ? 0.3 * (effectiveGpuCount - 1) : 0;
  const overheadGB = baseOverhead + hiddenOverhead + multiGpuOverhead;

  const totalRequiredGB = weightsGB + kvCacheGB + overheadGB;
  const headroomGB = effectiveUsableMemoryGB - totalRequiredGB;
  const fits = headroomGB >= 0;
  const utilizationPct =
    effectiveUsableMemoryGB > 0
      ? Math.min(100, (totalRequiredGB / effectiveUsableMemoryGB) * 100)
      : 100;

  const formatCompat = isModelFormatCompatible(model, device);
  const supported = formatCompat.supported && fits;
  const unsupportedReason = !formatCompat.supported
    ? formatCompat.reason
    : !fits
      ? `Requires ${totalRequiredGB.toFixed(1)} GB (${Math.abs(headroomGB).toFixed(1)} GB over usable budget)`
      : undefined;

  // 4. Bandwidth-bound decode & compute-bound prefill estimation
  const multiGpuScalingTable: Record<number, number> = {
    1: 1.0,
    2: 1.78,
    3: 2.52,
    4: 3.25,
    6: 4.55,
    8: 5.8,
  };
  const gpuScale =
    multiGpuScalingTable[effectiveGpuCount] ?? Math.pow(effectiveGpuCount, 0.85);
  const totalBandwidthGBs = device.bandwidthGBs * gpuScale;
  const backendEfficiency =
    formatCompat.recommendedBackend === "mlx"
      ? 0.78
      : formatCompat.recommendedBackend === "vllm"
        ? 0.82
        : 0.74;

  const activeParamsB = Math.max(0.25, model.activeParametersB || model.parametersB);
  const activeWeightReadGB =
    (activeParamsB * 1e9 * (quantSpec.bpw / 8)) / BYTES_PER_GIB;
  const kvReadOverheadGB = kvCacheGB * 0.18;
  const perTokenMemTrafficGB = Math.max(0.15, activeWeightReadGB + kvReadOverheadGB);

  const estimatedDecodeTps =
    arch.archetype === "diffusion"
      ? 0
      : Number(
          Math.max(0.4, (totalBandwidthGBs * backendEfficiency) / perTokenMemTrafficGB).toFixed(
            1
          )
        );

  const effectiveTflops = device.fp16Tflops * gpuScale * (config.quantization === "FP16" ? 0.55 : 0.72);
  const estimatedPrefillTps =
    arch.archetype === "diffusion"
      ? 0
      : Number(
          Math.max(12, (effectiveTflops * 1000) / Math.max(1, 2 * activeParamsB)).toFixed(
            0
          )
        );

  // 5. Max context fitting on this device
  const remainingForKvGB = Math.max(0, effectiveUsableMemoryGB - weightsGB - overheadGB);
  const maxContextRaw =
    arch.archetype === "diffusion" || kvCacheBytesPerToken <= 0
      ? 0
      : Math.floor(
          (remainingForKvGB * BYTES_PER_GIB) /
            (kvCacheBytesPerToken * Math.max(1, config.batchSize))
        );
  const maxContextOnDevice = Math.min(
    arch.maxContextLength,
    Math.max(0, Math.floor(maxContextRaw / 256) * 256)
  );

  return {
    weightsGB,
    kvCacheGB,
    overheadGB,
    totalRequiredGB,
    effectiveUsableMemoryGB,
    utilizationPct,
    headroomGB,
    fits,
    supported,
    unsupportedReason,
    estimatedDecodeTps,
    estimatedPrefillTps,
    maxContextOnDevice,
    recommendedBackend: formatCompat.recommendedBackend,
    weightsSource,
  };
}

export interface CompatibleModelResult {
  model: HFModelSummary;
  estimate: InferenceEstimate;
}

export function filterSupportedModelsForDevice(
  models: HFModelSummary[],
  device: HardwareDevice,
  config: InferenceConfig
): CompatibleModelResult[] {
  const results: CompatibleModelResult[] = [];
  for (const model of models) {
    const estimate = calculateInference(model, device, config);
    if (estimate.supported) {
      results.push({ model, estimate });
    }
  }
  return results;
}

export function generateLocalAIYaml(
  model: HFModelSummary | HFModelDetail,
  config: InferenceConfig,
  estimate: InferenceEstimate
): string {
  const slug = model.name
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-|-$/g, "");

  let modelSourceRef = `huggingface://${model.id}`;
  if (isDetailModel(model) && model.quantVariants.length > 0) {
    const exactVariant = model.quantVariants.find(
      (v) => v.mappedQuant === config.quantization
    );
    if (exactVariant && exactVariant.files.length > 0) {
      modelSourceRef = `huggingface://${model.id}/${exactVariant.files[0]}`;
    }
  }

  const lines = [
    `name: ${slug}`,
    `backend: ${estimate.recommendedBackend}`,
    `parameters:`,
    `  model: ${modelSourceRef}`,
    `context_size: ${config.contextLength}`,
    `gpu_layers: 999`,
    `f16: ${config.kvCachePrecision === "FP16" ? "true" : "false"}`,
    `mmap: true`,
  ];

  if (estimate.recommendedBackend === "llama-cpp") {
    lines.push(`flash_attention: ${config.flashAttention ? "true" : "false"}`);
    if (config.kvCachePrecision !== "FP16") {
      const cacheType = config.kvCachePrecision === "Q8_0" ? "q8_0" : "q4_0";
      lines.push(`cache_type_k: ${cacheType}`);
      lines.push(`cache_type_v: ${cacheType}`);
    }
  }

  if (estimate.recommendedBackend === "vllm" && config.gpuCount > 1) {
    lines.push(`tensor_parallel_size: ${config.gpuCount}`);
  }

  return lines.join("\n");
}
