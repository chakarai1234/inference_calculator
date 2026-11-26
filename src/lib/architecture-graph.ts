import type {
  ArchitectureArchetype,
  HFModelDetail,
  HFModelSummary,
  InferenceConfig,
  ModelArchitectureConfig,
} from "@/types/inference";
import { inferArchitectureFallback, KV_CACHE_BYTES, QUANT_MAP } from "@/lib/inference-math";

export interface StageSubMatrix {
  name: string;
  shape: string;
  params: string;
}

export interface ArchitectureStageNode {
  id: string;
  label: string;
  shortLabel: string;
  category:
    | "input"
    | "embedding"
    | "rope"
    | "attention"
    | "kv-cache"
    | "ffn"
    | "moe-router"
    | "moe-expert"
    | "norm"
    | "head"
    | "output"
    | "vision"
    | "diffusion";
  colorHex: string;
  accentHex: string;
  position: [number, number, number];
  dimensions: [number, number, number];
  tensorShape: string;
  parameterCountLabel: string;
  memoryLabel: string;
  description: string;
  formula: string;
  subMatrices?: StageSubMatrix[];
  layerIndex?: number;
  layerStart?: number;
  layerEnd?: number;
  expertIndex?: number;
  isActiveExpert?: boolean;
}

export interface LayerSubStep {
  id: string;
  stepNumber: number;
  name: string;
  tensorShape: string;
  paramsLabel: string;
  memoryMBLabel: string;
  role: string;
  colorHex: string;
}

export interface LayerDetailBreakdown {
  layerNumber: number;
  totalLayers: number;
  depthRole: string;
  totalLayerParams: number;
  activeLayerParams: number;
  totalLayerParamsLabel: string;
  activeLayerParamsLabel: string;
  totalLayerMemoryMB: number;
  activeLayerMemoryMB: number;
  kvCacheThisLayerMB: number;
  layerFlopsPerTokenG: number;
  attnSharePct: number;
  ffnSharePct: number;
  subSteps: LayerSubStep[];
}

export interface ComponentBreakdownItem {
  id: string;
  name: string;
  colorHex: string;
  params: number;
  paramsLabel: string;
  pctOfTotal: number;
  memoryGB: number;
  detail: string;
}

export interface ArchitectureDeepMetrics {
  totalFlopsPerTokenG: number;
  arithmeticIntensity: number;
  gqaCompressionFactor: number;
  kvMemoryAtContextGB: number;
  kvSavedByGqaPct: number;
  embeddingSharePct: number;
  ffnSharePct: number;
  attentionSharePct: number;
}

export interface ArchitectureBlueprint {
  archetype: ArchitectureArchetype;
  modelId: string;
  architecture: ModelArchitectureConfig;
  stages: ArchitectureStageNode[];
  visualLayerBlocks: number;
  layersPerVisualBlock: number;
  phaseSteps: { id: string; title: string; detail: string; colorHex: string }[];
  componentBreakdown: ComponentBreakdownItem[];
  deepMetrics: ArchitectureDeepMetrics;
  metrics: {
    gqaRatioLabel: string;
    headDim: number;
    kvPerTokenKB: number;
    totalParamsLabel: string;
    activeParamsLabel: string;
    quantLabel: string;
  };
}

export const STAGE_COLORS: Record<
  ArchitectureStageNode["category"],
  { main: string; accent: string }
> = {
  input: { main: "#0891B2", accent: "#22D3EE" },
  embedding: { main: "#2563EB", accent: "#60A5FA" },
  rope: { main: "#D97706", accent: "#FBBF24" },
  attention: { main: "#0284C7", accent: "#38BDF8" },
  "kv-cache": { main: "#22C55E", accent: "#4ADE80" },
  ffn: { main: "#EA580C", accent: "#FB923C" },
  "moe-router": { main: "#DB2777", accent: "#F472B6" },
  "moe-expert": { main: "#F59E0B", accent: "#FCD34D" },
  vision: { main: "#0D9488", accent: "#2DD4BF" },
  norm: { main: "#64748B", accent: "#94A3B8" },
  head: { main: "#4F46E5", accent: "#818CF8" },
  output: { main: "#16A34A", accent: "#4ADE80" },
  diffusion: { main: "#DB2777", accent: "#F472B6" },
};

function formatParamCountShort(count: number): string {
  if (count >= 1e9) return `${(count / 1e9).toFixed(2)}B params`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M params`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(0)}K params`;
  return `${Math.round(count)} params`;
}

function formatMB(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  if (mb >= 10) return `${mb.toFixed(0)} MB`;
  return `${mb.toFixed(1)} MB`;
}

export function buildSingleLayerBreakdown(
  model: HFModelSummary | HFModelDetail,
  config: InferenceConfig,
  requestedLayer: number
): LayerDetailBreakdown {
  const arch = inferArchitectureFallback(model);
  const quantSpec = QUANT_MAP[config.quantization] ?? QUANT_MAP.Q4_K_M;
  const bytesPerParam = quantSpec.bpw / 8;
  const totalLayers = Math.max(1, arch.numLayers);
  const layerNumber = Math.max(1, Math.min(totalLayers, Math.round(requestedLayer)));

  const ratio = layerNumber / totalLayers;
  const depthRole =
    ratio <= 0.25
      ? "Early Syntactic & Token Local Mixing"
      : ratio <= 0.75
        ? "Mid-Stack Semantic Reasoning & Retrieval"
        : "Deep Conceptual Synthesis & Logit Calibration";

  const headDim = Math.max(
    64,
    Math.round(arch.hiddenSize / Math.max(1, arch.numAttentionHeads))
  );
  const qDim = arch.numAttentionHeads * headDim;
  const kvDim = arch.numKeyValueHeads * headDim;
  const qProjParams = arch.hiddenSize * qDim;
  const kProjParams = arch.hiddenSize * kvDim;
  const vProjParams = arch.hiddenSize * kvDim;
  const oProjParams = qDim * arch.hiddenSize;
  const attnTotalParams = qProjParams + kProjParams + vProjParams + oProjParams;

  const isMoE = arch.archetype === "moe-transformer" && (arch.numExperts ?? 1) > 1;
  const expertCount = isMoE ? (arch.numExperts ?? 8) : 1;
  const activeExperts = isMoE ? (arch.numActiveExperts ?? 2) : 1;
  const singleExpertFfnParams = 3 * arch.hiddenSize * arch.intermediateSize;
  const routerParams = isMoE ? arch.hiddenSize * expertCount : 0;
  const ffnTotalParams = singleExpertFfnParams * expertCount + routerParams;
  const ffnActiveParams = singleExpertFfnParams * activeExperts + routerParams;
  const normParams = arch.hiddenSize * 2;

  const totalLayerParams = attnTotalParams + ffnTotalParams + normParams;
  const activeLayerParams = attnTotalParams + ffnActiveParams + normParams;

  const totalLayerMemoryMB = Number(
    ((totalLayerParams * bytesPerParam) / (1024 * 1024)).toFixed(2)
  );
  const activeLayerMemoryMB = Number(
    ((activeLayerParams * bytesPerParam) / (1024 * 1024)).toFixed(2)
  );

  const kvElementBytes = KV_CACHE_BYTES[config.kvCachePrecision] ?? 2;
  const kvCacheThisLayerMB = Number(
    (
      (2 * arch.numKeyValueHeads * headDim * config.contextLength * config.batchSize * kvElementBytes) /
      (1024 * 1024)
    ).toFixed(2)
  );

  const layerFlopsPerTokenG = Number(((2 * activeLayerParams) / 1e9).toFixed(3));
  const attnSharePct = Number(
    ((attnTotalParams / Math.max(1, totalLayerParams)) * 100).toFixed(1)
  );
  const ffnSharePct = Number(Math.max(0, 100 - attnSharePct).toFixed(1));

  const subSteps: LayerSubStep[] = [
    {
      id: "in-norm",
      stepNumber: 1,
      name: "Pre-Attention RMSNorm",
      tensorShape: `[B=${config.batchSize}, S, ${arch.hiddenSize}]`,
      paramsLabel: formatParamCountShort(arch.hiddenSize),
      memoryMBLabel: `${((arch.hiddenSize * 2) / (1024 * 1024)).toFixed(2)} MB`,
      role: "Stabilizes activation magnitudes entering attention.",
      colorHex: STAGE_COLORS.norm.main,
    },
    {
      id: "wq",
      stepNumber: 2,
      name: "Query Projection (W_q) + RoPE",
      tensorShape: `[${arch.hiddenSize}, ${qDim}] (${arch.numAttentionHeads} heads)`,
      paramsLabel: formatParamCountShort(qProjParams),
      memoryMBLabel: formatMB(qProjParams * bytesPerParam),
      role: "Projects residual vector into Q heads and applies rotary position angles.",
      colorHex: STAGE_COLORS.rope.main,
    },
    {
      id: "wkv",
      stepNumber: 3,
      name: "Key & Value Projections (W_k, W_v)",
      tensorShape: `2 × [${arch.hiddenSize}, ${kvDim}] (${arch.numKeyValueHeads} KV heads)`,
      paramsLabel: formatParamCountShort(kProjParams + vProjParams),
      memoryMBLabel: formatMB((kProjParams + vProjParams) * bytesPerParam),
      role: `GQA compressed KV projections; appends ${kvCacheThisLayerMB} MB to Layer ${layerNumber} KV Cache.`,
      colorHex: STAGE_COLORS["kv-cache"].main,
    },
    {
      id: "attn-core",
      stepNumber: 4,
      name: "Scaled Dot-Product Attention + W_o",
      tensorShape: `softmax(Q·Kᵀ/√${headDim})·V → [${qDim}, ${arch.hiddenSize}]`,
      paramsLabel: formatParamCountShort(oProjParams),
      memoryMBLabel: formatMB(oProjParams * bytesPerParam),
      role: "Mixes relevant context tokens across heads and projects back to residual stream.",
      colorHex: STAGE_COLORS.attention.main,
    },
    {
      id: "post-norm",
      stepNumber: 5,
      name: "Residual Add #1 & Post-Attn RMSNorm",
      tensorShape: `[B=${config.batchSize}, S, ${arch.hiddenSize}]`,
      paramsLabel: formatParamCountShort(arch.hiddenSize),
      memoryMBLabel: `${((arch.hiddenSize * 2) / (1024 * 1024)).toFixed(2)} MB`,
      role: "Preserves clean gradient/signal highway before MLP transformation.",
      colorHex: STAGE_COLORS.norm.main,
    },
  ];

  if (isMoE) {
    subSteps.push(
      {
        id: "moe-gate",
        stepNumber: 6,
        name: `MoE Top-${activeExperts} Gating Router`,
        tensorShape: `[${arch.hiddenSize}, ${expertCount} experts]`,
        paramsLabel: formatParamCountShort(routerParams),
        memoryMBLabel: formatMB(routerParams * bytesPerParam),
        role: `Routes each token only to its top ${activeExperts} specialized expert blades.`,
        colorHex: STAGE_COLORS["moe-router"].main,
      },
      {
        id: "moe-experts",
        stepNumber: 7,
        name: `SwiGLU Experts (${activeExperts} active / ${expertCount} total)`,
        tensorShape: `[${arch.hiddenSize} → ${arch.intermediateSize} → ${arch.hiddenSize}]`,
        paramsLabel: `${formatParamCountShort(ffnActiveParams)} active (${formatParamCountShort(ffnTotalParams)} total)`,
        memoryMBLabel: `${formatMB(ffnActiveParams * bytesPerParam)} active / ${formatMB(ffnTotalParams * bytesPerParam)} VRAM`,
        role: "Only active experts are read during token decode; dormant experts stay idle in VRAM.",
        colorHex: STAGE_COLORS.ffn.main,
      }
    );
  } else {
    subSteps.push({
      id: "swiglu-mlp",
      stepNumber: 6,
      name: "SwiGLU FFN (W_gate, W_up, W_down)",
      tensorShape: `[${arch.hiddenSize} → ${arch.intermediateSize} → ${arch.hiddenSize}]`,
      paramsLabel: formatParamCountShort(ffnTotalParams),
      memoryMBLabel: formatMB(ffnTotalParams * bytesPerParam),
      role: "Expands residual representations via gated SiLU non-linearity for factual & reasoning recall.",
      colorHex: STAGE_COLORS.ffn.main,
    });
  }

  subSteps.push({
    id: "residual-out",
    stepNumber: isMoE ? 8 : 7,
    name:
      layerNumber < totalLayers
        ? `Residual Add #2 → Layer ${layerNumber + 1}`
        : "Residual Add #2 → Final RMSNorm",
    tensorShape: `[B=${config.batchSize}, S, ${arch.hiddenSize}]`,
    paramsLabel: "Activation stream",
    memoryMBLabel: "Zero extra weights",
    role: "Sums transformed FFN output with residual stream and forwards to next stage.",
    colorHex: STAGE_COLORS.output.main,
  });

  return {
    layerNumber,
    totalLayers,
    depthRole,
    totalLayerParams,
    activeLayerParams,
    totalLayerParamsLabel: formatParamCountShort(totalLayerParams),
    activeLayerParamsLabel: formatParamCountShort(activeLayerParams),
    totalLayerMemoryMB,
    activeLayerMemoryMB,
    kvCacheThisLayerMB,
    layerFlopsPerTokenG,
    attnSharePct,
    ffnSharePct,
    subSteps,
  };
}

export function buildArchitectureBlueprint(
  model: HFModelSummary | HFModelDetail,
  config: InferenceConfig
): ArchitectureBlueprint {
  const arch = inferArchitectureFallback(model);
  const quantSpec = QUANT_MAP[config.quantization] ?? QUANT_MAP.Q4_K_M;
  const bytesPerParam = quantSpec.bpw / 8;

  const headDim = Math.max(
    64,
    Math.round(arch.hiddenSize / Math.max(1, arch.numAttentionHeads))
  );
  const gqaRatio = Math.max(
    1,
    Math.round(arch.numAttentionHeads / Math.max(1, arch.numKeyValueHeads))
  );
  const gqaRatioLabel =
    gqaRatio === 1
      ? `MHA (${arch.numAttentionHeads}:${arch.numKeyValueHeads})`
      : arch.numKeyValueHeads === 1
        ? `MQA (${arch.numAttentionHeads}:1)`
        : `GQA (${arch.numAttentionHeads}:${arch.numKeyValueHeads}, ${gqaRatio}:1)`;

  const kvElementBytes = KV_CACHE_BYTES[config.kvCachePrecision] ?? 2;
  const kvPerTokenKB =
    (2 * arch.numLayers * arch.numKeyValueHeads * headDim * kvElementBytes) / 1024;
  const kvMemoryAtContextGB =
    (kvPerTokenKB * 1024 * config.contextLength * config.batchSize) / (1024 * 1024 * 1024);
  const kvSavedByGqaPct =
    arch.numAttentionHeads > 0
      ? Number(((1 - arch.numKeyValueHeads / arch.numAttentionHeads) * 100).toFixed(1))
      : 0;

  const activeParams = Math.max(0.25, model.activeParametersB || model.parametersB) * 1e9;
  const totalFlopsPerTokenG = Number(((2 * activeParams) / 1e9).toFixed(2));
  const activeMemReadBytes = activeParams * bytesPerParam;
  const arithmeticIntensity = Number(
    ((2 * activeParams) / Math.max(1, activeMemReadBytes)).toFixed(2)
  );

  const stages: ArchitectureStageNode[] = [];

  if (arch.archetype === "diffusion") {
    const condParams = model.parametersB * 0.35e9;
    const ditParams = model.parametersB * 0.6e9;
    const vaeParams = 85e6;
    const sumParams = condParams + ditParams + vaeParams;

    stages.push(
      {
        id: "prompt-encoder",
        label: "Text Conditioning Encoder (CLIP / T5)",
        shortLabel: "TEXT COND",
        category: "vision",
        colorHex: STAGE_COLORS.vision.main,
        accentHex: STAGE_COLORS.vision.accent,
        position: [-9.5, 1.5, 0],
        dimensions: [2.0, 3.2, 2.2],
        tensorShape: `[B, 512, ${arch.hiddenSize}]`,
        parameterCountLabel: formatParamCountShort(condParams),
        memoryLabel: formatMB(condParams * bytesPerParam),
        description:
          "Encodes natural language prompt tokens into semantic conditioning embeddings projected into cross-attention blocks.",
        formula: "c = T5_CLIP(tokens)",
        subMatrices: [
          { name: "T5-XXL Embed", shape: `[32128, ${arch.hiddenSize}]`, params: formatParamCountShort(condParams * 0.6) },
          { name: "CLIP-L Pool", shape: `[77, 768]`, params: formatParamCountShort(condParams * 0.4) },
        ],
      },
      {
        id: "latent-noise",
        label: "Gaussian Latent Field",
        shortLabel: "LATENT z_T",
        category: "input",
        colorHex: STAGE_COLORS.input.main,
        accentHex: STAGE_COLORS.input.accent,
        position: [-9.5, -1.6, 0],
        dimensions: [1.8, 1.8, 1.8],
        tensorShape: `[B, 16, H/8, W/8]`,
        parameterCountLabel: "Activation buffer",
        memoryLabel: "240 MB",
        description:
          "Initial spatial latent tensor sampled from standard normal noise before iterative flow-matching / denoising.",
        formula: "z_T ~ N(0, I)",
      },
      {
        id: "dit-backbone",
        label: "DiT / UNet Iterative Denoising Core",
        shortLabel: "DiT BACKBONE",
        category: "diffusion",
        colorHex: STAGE_COLORS.diffusion.main,
        accentHex: STAGE_COLORS.diffusion.accent,
        position: [0, 0.2, 0],
        dimensions: [8.4, 4.2, 3.6],
        tensorShape: `[B, N_patches, ${arch.hiddenSize}] × ${arch.numLayers} blocks`,
        parameterCountLabel: formatParamCountShort(ditParams),
        memoryLabel: formatMB(ditParams * bytesPerParam),
        description:
          "Joint attention blocks fuse image patch latents with text conditioning and sinusoidal timestep embeddings across denoising steps.",
        formula: "z_{t-1} = Step(z_t, v_θ(z_t, t, c))",
        layerIndex: 0,
        layerStart: 1,
        layerEnd: arch.numLayers,
      },
      {
        id: "vae-decoder",
        label: "VAE Latent-to-Pixel Decoder",
        shortLabel: "VAE DECODER",
        category: "head",
        colorHex: STAGE_COLORS.head.main,
        accentHex: STAGE_COLORS.head.accent,
        position: [9.5, 0.2, 0],
        dimensions: [2.4, 3.4, 2.4],
        tensorShape: `[B, 3, H, W]`,
        parameterCountLabel: formatParamCountShort(vaeParams),
        memoryLabel: formatMB(vaeParams * 2),
        description:
          "Upsamples denoised 16-channel latents back into high-resolution RGB pixel space in a single forward pass.",
        formula: "x_0 = VAE.decode(z_0)",
      }
    );

    const componentBreakdown: ComponentBreakdownItem[] = [
      {
        id: "dit",
        name: "DiT / UNet Denoising Backbone",
        colorHex: STAGE_COLORS.diffusion.main,
        params: ditParams,
        paramsLabel: formatParamCountShort(ditParams),
        pctOfTotal: Number(((ditParams / sumParams) * 100).toFixed(1)),
        memoryGB: Number(((ditParams * bytesPerParam) / (1024 ** 3)).toFixed(2)),
        detail: `${arch.numLayers} joint MM-DiT transformer blocks`,
      },
      {
        id: "cond",
        name: "Text Conditioning Encoders",
        colorHex: STAGE_COLORS.vision.main,
        params: condParams,
        paramsLabel: formatParamCountShort(condParams),
        pctOfTotal: Number(((condParams / sumParams) * 100).toFixed(1)),
        memoryGB: Number(((condParams * bytesPerParam) / (1024 ** 3)).toFixed(2)),
        detail: "Dual CLIP + T5 semantic prompt projections",
      },
      {
        id: "vae",
        name: "VAE Spatial Autoencoder",
        colorHex: STAGE_COLORS.head.main,
        params: vaeParams,
        paramsLabel: formatParamCountShort(vaeParams),
        pctOfTotal: Number(((vaeParams / sumParams) * 100).toFixed(1)),
        memoryGB: Number(((vaeParams * 2) / (1024 ** 3)).toFixed(2)),
        detail: "16-channel latent spatial decoder (kept FP16)",
      },
    ];

    return {
      archetype: "diffusion",
      modelId: model.id,
      architecture: arch,
      stages,
      visualLayerBlocks: 4,
      layersPerVisualBlock: Math.max(1, Math.round(arch.numLayers / 4)),
      phaseSteps: [
        { id: "cond", title: "1. Prompt Encoding", detail: "Tokenize prompt through T5/CLIP into conditioning vectors", colorHex: STAGE_COLORS.vision.main },
        { id: "noise", title: "2. Latent Initialization", detail: "Seed compressed 16-ch latent spatial grid", colorHex: STAGE_COLORS.input.main },
        { id: "denoise", title: "3. Iterative DiT Flow", detail: "Cross-attend text + timestep across DiT blocks", colorHex: STAGE_COLORS.diffusion.main },
        { id: "vae", title: "4. VAE Reconstruction", detail: "Decode denoised latents into output image tensor", colorHex: STAGE_COLORS.head.main },
      ],
      componentBreakdown,
      deepMetrics: {
        totalFlopsPerTokenG,
        arithmeticIntensity,
        gqaCompressionFactor: 1,
        kvMemoryAtContextGB: 0,
        kvSavedByGqaPct: 0,
        embeddingSharePct: 35,
        ffnSharePct: 40,
        attentionSharePct: 25,
      },
      metrics: {
        gqaRatioLabel: "Joint MM-DiT Attention",
        headDim,
        kvPerTokenKB: 0,
        totalParamsLabel: `${model.parametersB.toFixed(1)}B`,
        activeParamsLabel: `${model.activeParametersB.toFixed(1)}B`,
        quantLabel: quantSpec.id,
      },
    };
  }

  // Transformer Pipeline (Dense, Sparse MoE, Multimodal VLM)
  const visualLayerBlocks = 4;
  const layersPerVisualBlock = Math.max(
    1,
    Math.round(arch.numLayers / visualLayerBlocks)
  );

  const embedParams = arch.vocabSize * arch.hiddenSize;
  const qProjParams = arch.hiddenSize * (arch.numAttentionHeads * headDim);
  const kProjParams = arch.hiddenSize * (arch.numKeyValueHeads * headDim);
  const vProjParams = arch.hiddenSize * (arch.numKeyValueHeads * headDim);
  const oProjParams = (arch.numAttentionHeads * headDim) * arch.hiddenSize;
  const attnParamsPerLayer = qProjParams + kProjParams + vProjParams + oProjParams;
  const expertCount = arch.numExperts ?? 1;
  const activeExperts = arch.numActiveExperts ?? 1;
  const ffnParamsPerLayer = 3 * arch.hiddenSize * arch.intermediateSize * expertCount;
  const normParamsTotal = (arch.numLayers * 2 + 1) * arch.hiddenSize;
  const lmHeadParams = arch.tieWordEmbeddings ? 0 : arch.vocabSize * arch.hiddenSize;

  const totalAttnParams = attnParamsPerLayer * arch.numLayers;
  const totalFfnParams = ffnParamsPerLayer * arch.numLayers;
  const totalStructuralSum = Math.max(
    1,
    embedParams + totalAttnParams + totalFfnParams + normParamsTotal + lmHeadParams
  );

  stages.push(
    {
      id: "token-input",
      label: "Token Stream & BPE Tokenizer",
      shortLabel: "TOKENS",
      category: "input",
      colorHex: STAGE_COLORS.input.main,
      accentHex: STAGE_COLORS.input.accent,
      position: [-14.2, 0.4, 0],
      dimensions: [1.3, 2.0, 1.8],
      tensorShape: `[B=${config.batchSize}, S≤${config.contextLength}]`,
      parameterCountLabel: "Discrete IDs",
      memoryLabel: "< 1 MB",
      description:
        "Maps input prompt strings and newly sampled autoregressive tokens into integer vocabulary indices.",
      formula: "x_ids ∈ Z^(B×S)",
    },
    {
      id: "embedding",
      label: "Token Embedding Lookup Matrix",
      shortLabel: "EMBEDDING",
      category: "embedding",
      colorHex: STAGE_COLORS.embedding.main,
      accentHex: STAGE_COLORS.embedding.accent,
      position: [-11.4, 0.4, 0],
      dimensions: [1.6, 3.8, 3.0],
      tensorShape: `[${arch.vocabSize.toLocaleString()}, ${arch.hiddenSize}]`,
      parameterCountLabel: formatParamCountShort(embedParams),
      memoryLabel: formatMB(embedParams * bytesPerParam),
      description:
        "Projects each token ID into the residual stream hidden dimension before entering the transformer stack.",
      formula: "h_0 = W_E[x_ids]",
      subMatrices: [
        {
          name: "W_embed",
          shape: `[${arch.vocabSize.toLocaleString()}, ${arch.hiddenSize}]`,
          params: formatParamCountShort(embedParams),
        },
      ],
    },
    {
      id: "rope-ring",
      label: "Rotary Position Embedding (RoPE)",
      shortLabel: "RoPE θ",
      category: "rope",
      colorHex: STAGE_COLORS.rope.main,
      accentHex: STAGE_COLORS.rope.accent,
      position: [-8.6, 0.4, 0],
      dimensions: [1.1, 3.4, 3.4],
      tensorShape: `[S, d_head=${headDim}] θ=${Math.round(arch.ropeTheta).toLocaleString()}`,
      parameterCountLabel: "Cos/Sin frequencies",
      memoryLabel: "< 2 MB",
      description:
        "Rotates Query & Key vector pairs in complex subspaces so attention naturally encodes relative token distance across long context.",
      formula: "q_m = R(m, θ) · q",
    }
  );

  let visionParams = 0;
  if (arch.hasVisionEncoder) {
    const vHidden = arch.visionHiddenSize ?? 1152;
    const vLayers = arch.visionLayers ?? 24;
    visionParams = vLayers * 12 * vHidden * vHidden;
    stages.push({
      id: "vision-encoder",
      label: "Vision Patch ViT Encoder + Projector",
      shortLabel: "VISION ViT",
      category: "vision",
      colorHex: STAGE_COLORS.vision.main,
      accentHex: STAGE_COLORS.vision.accent,
      position: [-11.4, 4.6, -2.4],
      dimensions: [3.2, 2.0, 2.2],
      tensorShape: `[N_patches, ${vHidden}] → [${arch.hiddenSize}]`,
      parameterCountLabel: formatParamCountShort(visionParams),
      memoryLabel: formatMB(visionParams * bytesPerParam),
      description:
        "Splits images into visual patches through ViT layers and projects visual tokens straight into the LLM embedding space.",
      formula: "h_vis = Proj(ViT(Patches(I)))",
    });
  }

  const stackStartX = -5.2;
  const blockSpacing = 3.5;

  for (let i = 0; i < visualLayerBlocks; i++) {
    const x = stackStartX + i * blockSpacing;
    const startLayer = i * layersPerVisualBlock + 1;
    const endLayer =
      i === visualLayerBlocks - 1
        ? arch.numLayers
        : Math.min(arch.numLayers, (i + 1) * layersPerVisualBlock);
    const spanCount = Math.max(1, endLayer - startLayer + 1);
    const layerRangeLabel = startLayer === endLayer ? `L${startLayer}` : `L${startLayer}–L${endLayer}`;

    stages.push({
      id: `attn-block-${i}`,
      label: `Self-Attention Core (${layerRangeLabel})`,
      shortLabel: `ATTN ${layerRangeLabel}`,
      category: "attention",
      colorHex: STAGE_COLORS.attention.main,
      accentHex: STAGE_COLORS.attention.accent,
      position: [x, 1.5, 0],
      dimensions: [2.2, 2.2, 2.6],
      tensorShape: `${arch.numAttentionHeads}Q / ${arch.numKeyValueHeads}KV · d=${headDim}`,
      parameterCountLabel: formatParamCountShort(attnParamsPerLayer * spanCount),
      memoryLabel: formatMB(attnParamsPerLayer * spanCount * bytesPerParam),
      description: `Computes causal scaled dot-product attention (${gqaRatioLabel}) across ${spanCount} layers with RMSNorm pre-normalization and residual bypass.`,
      formula: "Attn(Q,K,V) = softmax(QK^T / √d_k)V",
      subMatrices: [
        { name: "W_q (per layer)", shape: `[${arch.hiddenSize}, ${arch.numAttentionHeads * headDim}]`, params: formatParamCountShort(qProjParams) },
        { name: "W_k (per layer)", shape: `[${arch.hiddenSize}, ${arch.numKeyValueHeads * headDim}]`, params: formatParamCountShort(kProjParams) },
        { name: "W_v (per layer)", shape: `[${arch.hiddenSize}, ${arch.numKeyValueHeads * headDim}]`, params: formatParamCountShort(vProjParams) },
        { name: "W_o (per layer)", shape: `[${arch.numAttentionHeads * headDim}, ${arch.hiddenSize}]`, params: formatParamCountShort(oProjParams) },
      ],
      layerIndex: i,
      layerStart: startLayer,
      layerEnd: endLayer,
    });

    if (arch.archetype === "moe-transformer" && expertCount > 1) {
      stages.push({
        id: `moe-router-${i}`,
        label: `MoE Top-${activeExperts} Gating Router (${layerRangeLabel})`,
        shortLabel: `ROUTER ${activeExperts}/${expertCount}`,
        category: "moe-router",
        colorHex: STAGE_COLORS["moe-router"].main,
        accentHex: STAGE_COLORS["moe-router"].accent,
        position: [x, -1.0, 0],
        dimensions: [1.6, 1.1, 1.6],
        tensorShape: `[B×S, ${arch.hiddenSize}] → Top-${activeExperts}(${expertCount})`,
        parameterCountLabel: formatParamCountShort(arch.hiddenSize * expertCount * spanCount),
        memoryLabel: formatMB(arch.hiddenSize * expertCount * spanCount * bytesPerParam),
        description: `Lightweight gating linear layer scores all ${expertCount} experts per token and dispatches activations only to the top ${activeExperts}.`,
        formula: `g(x) = TopK(softmax(W_g · x), k=${activeExperts})`,
        layerIndex: i,
        layerStart: startLayer,
        layerEnd: endLayer,
      });

      const displayedExperts = Math.min(8, expertCount);
      for (let e = 0; e < displayedExperts; e++) {
        const angle = (e / displayedExperts) * Math.PI * 2;
        const radius = 1.65;
        const isActive = e < Math.min(displayedExperts, activeExperts);
        stages.push({
          id: `moe-expert-${i}-${e}`,
          label: `Expert FFN #${e + 1} (${layerRangeLabel})`,
          shortLabel: `EXP ${e + 1}`,
          category: "moe-expert",
          colorHex: isActive ? STAGE_COLORS["moe-expert"].main : "#64748B",
          accentHex: isActive ? STAGE_COLORS["moe-expert"].accent : "#94A3B8",
          position: [
            x + Math.cos(angle) * 0.42,
            -2.75,
            Math.sin(angle) * radius,
          ],
          dimensions: [0.65, 1.2, 0.52],
          tensorShape: `[${arch.hiddenSize} → ${arch.intermediateSize} → ${arch.hiddenSize}]`,
          parameterCountLabel: formatParamCountShort(
            3 * arch.hiddenSize * arch.intermediateSize * spanCount
          ),
          memoryLabel: formatMB(
            3 * arch.hiddenSize * arch.intermediateSize * spanCount * bytesPerParam
          ),
          description: isActive
            ? `Active SwiGLU expert blade selected by the router for the current token step.`
            : `Dormant expert weights resident in VRAM, skipped on this forward step to boost tok/s.`,
          formula: "FFN(x) = W_down(SiLU(W_gate x) ⊙ W_up x)",
          subMatrices: [
            { name: "W_gate", shape: `[${arch.hiddenSize}, ${arch.intermediateSize}]`, params: formatParamCountShort(arch.hiddenSize * arch.intermediateSize) },
            { name: "W_up", shape: `[${arch.hiddenSize}, ${arch.intermediateSize}]`, params: formatParamCountShort(arch.hiddenSize * arch.intermediateSize) },
            { name: "W_down", shape: `[${arch.intermediateSize}, ${arch.hiddenSize}]`, params: formatParamCountShort(arch.intermediateSize * arch.hiddenSize) },
          ],
          layerIndex: i,
          layerStart: startLayer,
          layerEnd: endLayer,
          expertIndex: e,
          isActiveExpert: isActive,
        });
      }
    } else {
      stages.push({
        id: `ffn-block-${i}`,
        label: `SwiGLU MLP Block (${layerRangeLabel})`,
        shortLabel: `FFN ${layerRangeLabel}`,
        category: "ffn",
        colorHex: STAGE_COLORS.ffn.main,
        accentHex: STAGE_COLORS.ffn.accent,
        position: [x, -1.65, 0],
        dimensions: [2.2, 2.0, 2.6],
        tensorShape: `[${arch.hiddenSize} → ${arch.intermediateSize} → ${arch.hiddenSize}]`,
        parameterCountLabel: formatParamCountShort(ffnParamsPerLayer * spanCount),
        memoryLabel: formatMB(ffnParamsPerLayer * spanCount * bytesPerParam),
        description:
          "Dense gated feed-forward network (Gate, Up, and Down projections) performing non-linear factual recall and representation mixing.",
        formula: "FFN(x) = W_down(SiLU(W_gate x) ⊙ W_up x)",
        subMatrices: [
          { name: "W_gate (per layer)", shape: `[${arch.hiddenSize}, ${arch.intermediateSize}]`, params: formatParamCountShort(arch.hiddenSize * arch.intermediateSize) },
          { name: "W_up (per layer)", shape: `[${arch.hiddenSize}, ${arch.intermediateSize}]`, params: formatParamCountShort(arch.hiddenSize * arch.intermediateSize) },
          { name: "W_down (per layer)", shape: `[${arch.intermediateSize}, ${arch.hiddenSize}]`, params: formatParamCountShort(arch.intermediateSize * arch.hiddenSize) },
        ],
        layerIndex: i,
        layerStart: startLayer,
        layerEnd: endLayer,
      });
    }
  }

  const kvTotalBytes =
    2 *
    arch.numLayers *
    arch.numKeyValueHeads *
    headDim *
    config.contextLength *
    config.batchSize *
    kvElementBytes;

  stages.push({
    id: "kv-cache-bank",
    label: "KV Cache Memory Bank",
    shortLabel: "KV CACHE",
    category: "kv-cache",
    colorHex: STAGE_COLORS["kv-cache"].main,
    accentHex: STAGE_COLORS["kv-cache"].accent,
    position: [0.0, 1.5, -4.4],
    dimensions: [13.2, 1.7, 1.5],
    tensorShape: `[2, ${arch.numLayers}, ${arch.numKeyValueHeads}, ${config.contextLength}, ${headDim}]`,
    parameterCountLabel: `${kvPerTokenKB.toFixed(1)} KB / token`,
    memoryLabel: formatMB(kvTotalBytes),
    description:
      "Persistent GPU/Unified memory tensor caching historical Key and Value projections across layers to avoid quadratic recomputation during decode.",
    formula: "K_cache, V_cache ← concat(past, new)",
  });

  const lmHeadDisplayParams = arch.vocabSize * arch.hiddenSize;
  stages.push(
    {
      id: "final-norm",
      label: "Final RMSNorm",
      shortLabel: "RMSNORM",
      category: "norm",
      colorHex: STAGE_COLORS.norm.main,
      accentHex: STAGE_COLORS.norm.accent,
      position: [8.6, 0.4, 0],
      dimensions: [0.7, 3.2, 2.6],
      tensorShape: `[B, ${arch.hiddenSize}]`,
      parameterCountLabel: formatParamCountShort(arch.hiddenSize),
      memoryLabel: "< 1 MB",
      description:
        "Normalizes final residual stream hidden states prior to vocabulary logit projection.",
      formula: "y = x / √(mean(x²) + ε) · γ",
    },
    {
      id: "lm-head",
      label: "LM Head Unembedding Projection",
      shortLabel: "LM HEAD",
      category: "head",
      colorHex: STAGE_COLORS.head.main,
      accentHex: STAGE_COLORS.head.accent,
      position: [11.2, 0.4, 0],
      dimensions: [1.7, 3.8, 3.0],
      tensorShape: `[${arch.hiddenSize} × ${arch.vocabSize.toLocaleString()}]`,
      parameterCountLabel: arch.tieWordEmbeddings
        ? "Tied with Embed"
        : formatParamCountShort(lmHeadDisplayParams),
      memoryLabel: arch.tieWordEmbeddings ? "Shared" : formatMB(lmHeadDisplayParams * bytesPerParam),
      description:
        "Projects final hidden vector across the full token vocabulary to compute unnormalized logits.",
      formula: "logits = h_L · W_lm^T",
      subMatrices: [
        {
          name: "W_lm_head",
          shape: `[${arch.hiddenSize}, ${arch.vocabSize.toLocaleString()}]`,
          params: arch.tieWordEmbeddings ? "Weight-tied" : formatParamCountShort(lmHeadDisplayParams),
        },
      ],
    },
    {
      id: "softmax-sampler",
      label: "Softmax & Autoregressive Sampler",
      shortLabel: "NEXT TOKEN",
      category: "output",
      colorHex: STAGE_COLORS.output.main,
      accentHex: STAGE_COLORS.output.accent,
      position: [14.2, 0.4, 0],
      dimensions: [1.5, 2.6, 2.0],
      tensorShape: `[B=${config.batchSize}, V=${arch.vocabSize.toLocaleString()}] → token_t+1`,
      parameterCountLabel: "Sampling stage",
      memoryLabel: `${((config.batchSize * arch.vocabSize * 4) / (1024 * 1024)).toFixed(1)} MB logits`,
      description:
        "Applies temperature / top-p / top-k softmax sampling and loops the chosen token back to the input stream for the next decode step.",
      formula: "x_{t+1} ~ softmax(logits / τ)",
    }
  );

  const componentBreakdown: ComponentBreakdownItem[] = [
    {
      id: "ffn",
      name:
        arch.archetype === "moe-transformer"
          ? `Sparse MoE Experts (${expertCount}× per layer)`
          : "SwiGLU FFN MLP Blocks",
      colorHex: STAGE_COLORS.ffn.main,
      params: totalFfnParams,
      paramsLabel: formatParamCountShort(totalFfnParams),
      pctOfTotal: Number(((totalFfnParams / totalStructuralSum) * 100).toFixed(1)),
      memoryGB: Number(((totalFfnParams * bytesPerParam) / (1024 ** 3)).toFixed(2)),
      detail: `${arch.numLayers} layers · W_gate, W_up, W_down [${arch.hiddenSize} ↔ ${arch.intermediateSize}]`,
    },
    {
      id: "attention",
      name: `Self-Attention (${gqaRatioLabel})`,
      colorHex: STAGE_COLORS.attention.main,
      params: totalAttnParams,
      paramsLabel: formatParamCountShort(totalAttnParams),
      pctOfTotal: Number(((totalAttnParams / totalStructuralSum) * 100).toFixed(1)),
      memoryGB: Number(((totalAttnParams * bytesPerParam) / (1024 ** 3)).toFixed(2)),
      detail: `${arch.numLayers} layers · ${arch.numAttentionHeads}Q / ${arch.numKeyValueHeads}KV · d_head=${headDim}`,
    },
    {
      id: "embed",
      name: "Token Embeddings",
      colorHex: STAGE_COLORS.embedding.main,
      params: embedParams,
      paramsLabel: formatParamCountShort(embedParams),
      pctOfTotal: Number(((embedParams / totalStructuralSum) * 100).toFixed(1)),
      memoryGB: Number(((embedParams * bytesPerParam) / (1024 ** 3)).toFixed(2)),
      detail: `${arch.vocabSize.toLocaleString()} vocab × ${arch.hiddenSize} d_model`,
    },
  ];

  if (!arch.tieWordEmbeddings && lmHeadParams > 0) {
    componentBreakdown.push({
      id: "head",
      name: "LM Head Unembedding",
      colorHex: STAGE_COLORS.head.main,
      params: lmHeadParams,
      paramsLabel: formatParamCountShort(lmHeadParams),
      pctOfTotal: Number(((lmHeadParams / totalStructuralSum) * 100).toFixed(1)),
      memoryGB: Number(((lmHeadParams * bytesPerParam) / (1024 ** 3)).toFixed(2)),
      detail: `Logit projection [${arch.hiddenSize} → ${arch.vocabSize.toLocaleString()}]`,
    });
  }

  if (visionParams > 0) {
    componentBreakdown.push({
      id: "vision",
      name: "Vision ViT Encoder",
      colorHex: STAGE_COLORS.vision.main,
      params: visionParams,
      paramsLabel: formatParamCountShort(visionParams),
      pctOfTotal: Number(((visionParams / (totalStructuralSum + visionParams)) * 100).toFixed(1)),
      memoryGB: Number(((visionParams * bytesPerParam) / (1024 ** 3)).toFixed(2)),
      detail: `${arch.visionLayers ?? 24} ViT blocks · ${arch.visionHiddenSize ?? 1152} dim`,
    });
  }

  componentBreakdown.push({
    id: "norms",
    name: "RMSNorm & Layer Scales",
    colorHex: STAGE_COLORS.norm.main,
    params: normParamsTotal,
    paramsLabel: formatParamCountShort(normParamsTotal),
    pctOfTotal: Number(((normParamsTotal / totalStructuralSum) * 100).toFixed(2)),
    memoryGB: Number(((normParamsTotal * 2) / (1024 ** 3)).toFixed(4)),
    detail: `${arch.numLayers * 2 + 1} normalization vectors (kept FP16)`,
  });

  const phaseSteps =
    arch.archetype === "moe-transformer"
      ? [
          { id: "embed", title: "1. Tokenize & Embed", detail: "Token IDs map into residual stream hidden states", colorHex: STAGE_COLORS.embedding.main },
          { id: "attn", title: "2. RoPE & Self-Attention", detail: `Apply rotary phase & ${gqaRatioLabel} attention`, colorHex: STAGE_COLORS.attention.main },
          { id: "kv", title: "3. KV Cache Append", detail: `Stream ${kvPerTokenKB.toFixed(0)} KB/tok into emerald memory bank`, colorHex: STAGE_COLORS["kv-cache"].main },
          { id: "moe", title: "4. Sparse MoE Routing", detail: `Router activates Top-${activeExperts} of ${expertCount} expert FFN blades`, colorHex: STAGE_COLORS["moe-router"].main },
          { id: "sample", title: "5. LM Head & Token Loop", detail: "Project logits and stream token back for decode", colorHex: STAGE_COLORS.output.main },
        ]
      : [
          { id: "embed", title: "1. Tokenize & Embed", detail: "Token IDs map into residual stream hidden states", colorHex: STAGE_COLORS.embedding.main },
          { id: "attn", title: "2. RoPE & Self-Attention", detail: `Apply rotary phase & ${gqaRatioLabel} attention`, colorHex: STAGE_COLORS.attention.main },
          { id: "kv", title: "3. KV Cache Append", detail: `Stream ${kvPerTokenKB.toFixed(0)} KB/tok into emerald memory bank`, colorHex: STAGE_COLORS["kv-cache"].main },
          { id: "ffn", title: "4. SwiGLU MLP Transform", detail: "Expand & gate features via dense FFN projections", colorHex: STAGE_COLORS.ffn.main },
          { id: "sample", title: "5. LM Head & Token Loop", detail: "Project logits and stream token back for decode", colorHex: STAGE_COLORS.output.main },
        ];

  return {
    archetype: arch.archetype,
    modelId: model.id,
    architecture: arch,
    stages,
    visualLayerBlocks,
    layersPerVisualBlock,
    phaseSteps,
    componentBreakdown,
    deepMetrics: {
      totalFlopsPerTokenG,
      arithmeticIntensity,
      gqaCompressionFactor: gqaRatio,
      kvMemoryAtContextGB: Number(kvMemoryAtContextGB.toFixed(3)),
      kvSavedByGqaPct,
      embeddingSharePct: Number(((embedParams / totalStructuralSum) * 100).toFixed(1)),
      ffnSharePct: Number(((totalFfnParams / totalStructuralSum) * 100).toFixed(1)),
      attentionSharePct: Number(((totalAttnParams / totalStructuralSum) * 100).toFixed(1)),
    },
    metrics: {
      gqaRatioLabel,
      headDim,
      kvPerTokenKB,
      totalParamsLabel: `${model.parametersB.toFixed(1)}B`,
      activeParamsLabel: `${model.activeParametersB.toFixed(1)}B`,
      quantLabel: quantSpec.id,
    },
  };
}
