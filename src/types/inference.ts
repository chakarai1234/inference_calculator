export type ThemeMode = "light" | "dark";

export type QuantizationLevel =
  | "FP16"
  | "FP8"
  | "Q8_0"
  | "Q6_K"
  | "Q5_K_M"
  | "Q4_K_M"
  | "IQ4_XS"
  | "Q3_K_M"
  | "Q2_K";

export type KVCachePrecision = "FP16" | "Q8_0" | "Q4_0";

export type ModelFormat = "gguf" | "safetensors" | "mlx" | "diffusers" | "unknown";

export type LocalAIBackend =
  | "llama-cpp"
  | "vllm"
  | "mlx"
  | "diffusers"
  | "transformers";

export type HardwareVendor =
  | "apple"
  | "nvidia-consumer"
  | "nvidia-datacenter"
  | "amd"
  | "intel";

export interface HardwareDevice {
  id: string;
  name: string;
  vendor: HardwareVendor;
  vendorLabel: string;
  architecture: string;
  vramGB: number;
  bandwidthGBs: number;
  fp16Tflops: number;
  fp8Tflops: number;
  int4Tops: number;
  memoryType: string;
  memoryBusBits: number;
  tdpWatts: number;
  interconnect: string;
  computeCores: string;
  cpuRamRecommendation: string;
  processNode: string;
  unifiedMemory: boolean;
  defaultUsableRatio: number;
  maxUsableRatio: number;
  supportsMultiGpu: boolean;
  maxGpuCount: number;
  supportedBackends: LocalAIBackend[];
  releaseYear: number;
  formFactor: string;
  notes: string;
}

export type ArchitectureArchetype =
  | "dense-transformer"
  | "moe-transformer"
  | "multimodal-vlm"
  | "diffusion";

export interface ModelArchitectureConfig {
  archetype: ArchitectureArchetype;
  source: "config.json" | "gguf" | "estimated";
  modelType: string;
  architectureName: string;
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  vocabSize: number;
  maxContextLength: number;
  ropeTheta: number;
  tieWordEmbeddings: boolean;
  numExperts?: number;
  numActiveExperts?: number;
  hasVisionEncoder?: boolean;
  visionLayers?: number;
  visionHiddenSize?: number;
  rawConfig?: Record<string, unknown>;
}

export interface ModelFileEntry {
  path: string;
  sizeBytes: number;
  format: ModelFormat;
  quantLabel?: string;
  mappedQuant?: QuantizationLevel;
  isShard?: boolean;
}

export interface QuantVariantEntry {
  quant: string;
  mappedQuant?: QuantizationLevel;
  totalBytes: number;
  files: string[];
}

export interface HFModelSummary {
  id: string;
  author: string;
  name: string;
  pipelineTag: string;
  libraryName: string;
  formats: ModelFormat[];
  primaryFormat: ModelFormat;
  parametersB: number;
  activeParametersB: number;
  isMoE: boolean;
  architectureHint: string;
  defaultContextLength: number;
  downloads: number;
  likes: number;
  gated: boolean;
  lastModified: string;
  tags: string[];
  estimatedDiskGBAtFP16: number;
  estimatedDiskGBAtQ4: number;
  exactBaseBytes?: number;
  embeddedConfig?: Partial<ModelArchitectureConfig>;
}

export interface HFModelDetail extends HFModelSummary {
  files: ModelFileEntry[];
  quantVariants: QuantVariantEntry[];
  totalRepoWeightBytes: number;
  architecture: ModelArchitectureConfig;
  license?: string;
  cardSummary?: string;
}

export interface InferenceConfig {
  quantization: QuantizationLevel;
  contextLength: number;
  batchSize: number;
  kvCachePrecision: KVCachePrecision;
  flashAttention: boolean;
  gpuCount: number;
  appleWiredRatio: number;
}

export interface InferenceEstimate {
  weightsGB: number;
  kvCacheGB: number;
  overheadGB: number;
  totalRequiredGB: number;
  effectiveUsableMemoryGB: number;
  utilizationPct: number;
  headroomGB: number;
  fits: boolean;
  supported: boolean;
  unsupportedReason?: string;
  estimatedDecodeTps: number;
  estimatedPrefillTps: number;
  maxContextOnDevice: number;
  recommendedBackend: LocalAIBackend;
  weightsSource: "exact-file" | "parameter-math";
}

export interface CatalogSnapshot {
  models: HFModelSummary[];
  lastRefreshedAt: string;
  nextScheduledRefreshAt: string;
  source: "live" | "cache" | "seed";
  count: number;
}

export interface QuantizationSpec {
  id: QuantizationLevel;
  label: string;
  bpw: number;
  qualityNote: string;
}
