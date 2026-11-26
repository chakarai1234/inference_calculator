import type {
  HFModelDetail,
  HFModelSummary,
  ModelArchitectureConfig,
  ModelFileEntry,
  ModelFormat,
  QuantizationLevel,
  QuantVariantEntry,
} from "@/types/inference";
import { inferArchitectureFallback } from "@/lib/inference-math";

const HF_API_BASE = "https://huggingface.co";
const GIB = 1024 * 1024 * 1024;
const DETAIL_TTL_MS = 10 * 60 * 1000;
const MAX_DETAIL_CACHE_ENTRIES = 40;

interface RawHFSibling {
  rfilename: string;
  size?: number;
  lfs?: { size?: number };
}

interface RawHFApiModel {
  _id?: string;
  id?: string;
  modelId?: string;
  author?: string;
  pipeline_tag?: string;
  library_name?: string;
  downloads?: number;
  likes?: number;
  gated?: boolean | string;
  lastModified?: string;
  tags?: string[];
  safetensors?: {
    total?: number;
    parameters?: Record<string, number>;
  };
  gguf?: {
    total?: number;
    architecture?: string;
    context_length?: number;
  };
  config?: Record<string, unknown>;
  siblings?: RawHFSibling[];
  cardData?: {
    license?: string | string[];
  };
}

interface RawTreeEntry {
  type: "file" | "directory";
  path: string;
  size?: number;
  lfs?: {
    size?: number;
    sha256?: string;
  };
}

const detailCache = new Map<string, { detail: HFModelDetail; expiresAt: number }>();

function setCachedDetail(repoId: string, detail: HFModelDetail): void {
  const now = Date.now();
  // Re-insert so Map iteration order reflects recency (LRU)
  if (detailCache.has(repoId)) {
    detailCache.delete(repoId);
  } else if (detailCache.size >= MAX_DETAIL_CACHE_ENTRIES) {
    // First sweep expired entries before evicting the oldest active entry
    for (const [key, entry] of detailCache.entries()) {
      if (entry.expiresAt <= now) {
        detailCache.delete(key);
      }
    }
    while (detailCache.size >= MAX_DETAIL_CACHE_ENTRIES) {
      const oldestKey = detailCache.keys().next().value;
      if (oldestKey === undefined) break;
      detailCache.delete(oldestKey);
    }
  }
  detailCache.set(repoId, { detail, expiresAt: now + DETAIL_TTL_MS });
}

export function getHFHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    Accept: "application/json",
    "User-Agent": "monolith-localai-calculator/1.0",
  };
  const token = process.env.HF_TOKEN?.trim();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

async function fetchJsonWithTimeout<T>(url: string, timeoutMs = 11000): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      headers: getHFHeaders(),
      signal: controller.signal,
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`HF HTTP ${res.status} for ${url}`);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

export function isValidRepoId(repoId: string): boolean {
  return /^[a-zA-Z0-9._-]+(\/[a-zA-Z0-9._-]+)?$/.test(repoId) && !repoId.includes("..");
}

function detectFormats(raw: RawHFApiModel): ModelFormat[] {
  const tags = (raw.tags ?? []).map((t) => t.toLowerCase());
  const lib = (raw.library_name ?? "").toLowerCase();
  const idLower = (raw.id ?? raw.modelId ?? "").toLowerCase();
  const formats: ModelFormat[] = [];

  if (raw.gguf || tags.includes("gguf") || lib === "gguf" || idLower.includes("gguf")) {
    formats.push("gguf");
  }
  if (lib === "mlx" || tags.includes("mlx") || idLower.startsWith("mlx-community/")) {
    formats.push("mlx");
  }
  if (
    lib === "diffusers" ||
    tags.includes("diffusers") ||
    raw.pipeline_tag === "text-to-image"
  ) {
    formats.push("diffusers");
  }
  if (
    raw.safetensors ||
    tags.includes("safetensors") ||
    lib === "transformers" ||
    formats.length === 0
  ) {
    if (!formats.includes("safetensors")) {
      formats.push("safetensors");
    }
  }
  return formats;
}

export function extractParamCounts(raw: RawHFApiModel): {
  parametersB: number;
  activeParametersB: number;
  isMoE: boolean;
} {
  const id = raw.id ?? raw.modelId ?? "";
  const idLower = id.toLowerCase();
  const tagsJoined = (raw.tags ?? []).join(" ").toLowerCase();

  let totalB = 0;

  if (raw.safetensors?.total && raw.safetensors.total > 10_000_000) {
    totalB = raw.safetensors.total / 1e9;
  } else if (raw.gguf?.total && raw.gguf.total > 10_000_000) {
    totalB = raw.gguf.total / 1e9;
  }

  let activeB = 0;
  let isMoE = false;

  // Check explicit active pattern like Qwen3-235B-A22B or Qwen3-30B-A3B
  const activePattern = id.match(/(\d+(?:\.\d+)?)\s*[bB][-_ ]*[aA](\d+(?:\.\d+)?)\s*[bB]/);
  if (activePattern) {
    const parsedTotal = parseFloat(activePattern[1]);
    const parsedActive = parseFloat(activePattern[2]);
    if (totalB <= 0) totalB = parsedTotal;
    activeB = parsedActive;
    isMoE = true;
  }

  // Check Mixtral style NxMB (e.g., 8x7B, 8x22B)
  const moeMulMatch = id.match(/(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*[bB]/i);
  if (moeMulMatch) {
    const experts = parseInt(moeMulMatch[1], 10);
    const expertSize = parseFloat(moeMulMatch[2]);
    if (totalB <= 0) {
      totalB = Number((experts * expertSize * 0.83).toFixed(1));
    }
    if (activeB <= 0) {
      activeB = Number((expertSize * 1.85).toFixed(1));
    }
    isMoE = true;
  }

  // DeepSeek V3 / R1 671B MoE (37B active)
  if (
    (idLower.includes("deepseek-r1") || idLower.includes("deepseek-v3")) &&
    !idLower.includes("distill")
  ) {
    if (totalB <= 0 || totalB > 600) {
      totalB = totalB > 0 ? totalB : 671;
      activeB = 37;
      isMoE = true;
    }
  }

  // Fallback regex for billion/million scale in ID or tags
  if (totalB <= 0) {
    const bMatch = `${id} ${tagsJoined}`.match(/(?:^|[^a-z0-9])(\d+(?:\.\d+)?)\s*b(?:[^a-z0-9]|$)/i);
    if (bMatch) {
      totalB = parseFloat(bMatch[1]);
    } else {
      const mMatch = `${id} ${tagsJoined}`.match(/(?:^|[^a-z0-9])(\d+(?:\.\d+)?)\s*m(?:[^a-z0-9]|$)/i);
      if (mMatch) {
        totalB = parseFloat(mMatch[1]) / 1000;
      }
    }
  }

  // Config-based fallback check for MoE experts
  const cfg = (raw.config?.text_config as Record<string, unknown> | undefined) ?? raw.config;
  const numExperts = Number(cfg?.num_local_experts ?? cfg?.n_routed_experts ?? 0);
  const numActive = Number(cfg?.num_experts_per_tok ?? 0);
  if (numExperts > 1 && numActive >= 1) {
    isMoE = true;
    if (activeB <= 0 && totalB > 0) {
      activeB = Number(
        Math.max(totalB * (numActive / numExperts) * 1.35, totalB * 0.08).toFixed(1)
      );
    }
  }

  if (totalB <= 0) {
    totalB = 7.0;
  }
  if (activeB <= 0) {
    activeB = totalB;
  }

  return {
    parametersB: Number(totalB.toFixed(2)),
    activeParametersB: Number(Math.min(totalB, activeB).toFixed(2)),
    isMoE,
  };
}

function extractEmbeddedConfig(raw: RawHFApiModel): Partial<ModelArchitectureConfig> | undefined {
  const cfgRoot = raw.config;
  if (!cfgRoot && !raw.gguf) return undefined;

  const textCfg =
    (cfgRoot?.text_config as Record<string, unknown> | undefined) ?? cfgRoot ?? {};
  const visionCfg = cfgRoot?.vision_config as Record<string, unknown> | undefined;

  const numLayers = Number(
    textCfg.num_hidden_layers ?? textCfg.n_layer ?? 0
  );
  const hiddenSize = Number(
    textCfg.hidden_size ?? textCfg.n_embd ?? textCfg.d_model ?? 0
  );
  const intermediateSize = Number(
    textCfg.intermediate_size ?? textCfg.ffn_dim ?? 0
  );
  const numAttentionHeads = Number(
    textCfg.num_attention_heads ?? textCfg.n_head ?? 0
  );
  const numKeyValueHeads = Number(
    textCfg.num_key_value_heads ?? numAttentionHeads ?? 0
  );
  const vocabSize = Number(textCfg.vocab_size ?? 0);
  const maxContextLength = Number(
    textCfg.max_position_embeddings ?? raw.gguf?.context_length ?? 0
  );
  const ropeTheta = Number(textCfg.rope_theta ?? 0);
  const numExperts = Number(
    textCfg.num_local_experts ?? textCfg.n_routed_experts ?? 0
  );
  const numActiveExperts = Number(textCfg.num_experts_per_tok ?? 0);

  const archArr = Array.isArray(cfgRoot?.architectures)
    ? (cfgRoot?.architectures as string[])
    : [];

  return {
    modelType: typeof textCfg.model_type === "string" ? textCfg.model_type : raw.gguf?.architecture,
    architectureName: archArr[0] ?? raw.gguf?.architecture,
    numLayers: numLayers > 0 ? numLayers : undefined,
    hiddenSize: hiddenSize > 0 ? hiddenSize : undefined,
    intermediateSize: intermediateSize > 0 ? intermediateSize : undefined,
    numAttentionHeads: numAttentionHeads > 0 ? numAttentionHeads : undefined,
    numKeyValueHeads: numKeyValueHeads > 0 ? numKeyValueHeads : undefined,
    vocabSize: vocabSize > 0 ? vocabSize : undefined,
    maxContextLength: maxContextLength > 0 ? maxContextLength : undefined,
    ropeTheta: ropeTheta > 0 ? ropeTheta : undefined,
    numExperts: numExperts > 1 ? numExperts : undefined,
    numActiveExperts: numActiveExperts >= 1 ? numActiveExperts : undefined,
    hasVisionEncoder: Boolean(visionCfg),
    visionLayers: visionCfg ? Number(visionCfg.num_hidden_layers ?? 24) : undefined,
    visionHiddenSize: visionCfg ? Number(visionCfg.hidden_size ?? 1152) : undefined,
  };
}

export function normalizeHFSummary(raw: RawHFApiModel): HFModelSummary | null {
  const id = raw.id ?? raw.modelId;
  if (!id || !id.includes("/")) return null;

  const formats = detectFormats(raw);
  const { parametersB, activeParametersB, isMoE } = extractParamCounts(raw);

  // Skip implausible toy/empty entries unless explicitly valid
  if (!Number.isFinite(parametersB) || parametersB <= 0.05) return null;

  const [author, ...rest] = id.split("/");
  const name = rest.join("/");
  const embedded = extractEmbeddedConfig(raw);
  const archHint =
    embedded?.architectureName ??
    embedded?.modelType ??
    raw.gguf?.architecture ??
    (raw.pipeline_tag === "text-to-image" ? "Diffusion" : "Transformer");

  const defaultCtx = embedded?.maxContextLength ?? raw.gguf?.context_length ?? 32768;
  const estFP16 = Number(((parametersB * 1e9 * 2) / GIB).toFixed(2));
  const estQ4 = Number(((parametersB * 1e9 * (4.85 / 8)) / GIB).toFixed(2));

  return {
    id,
    author: raw.author ?? author,
    name,
    pipelineTag: raw.pipeline_tag ?? "text-generation",
    libraryName: raw.library_name ?? formats[0],
    formats,
    primaryFormat: formats[0],
    parametersB,
    activeParametersB,
    isMoE,
    architectureHint: String(archHint),
    defaultContextLength: defaultCtx,
    downloads: raw.downloads ?? 0,
    likes: raw.likes ?? 0,
    gated: Boolean(raw.gated),
    lastModified: raw.lastModified ?? new Date().toISOString(),
    tags: (raw.tags ?? []).slice(0, 14),
    estimatedDiskGBAtFP16: estFP16,
    estimatedDiskGBAtQ4: estQ4,
    embeddedConfig: embedded,
  };
}

function buildExpandParams(): string {
  const fields = [
    "safetensors",
    "gguf",
    "config",
    "downloads",
    "likes",
    "tags",
    "pipeline_tag",
    "library_name",
    "lastModified",
    "gated",
    "author",
  ];
  return fields.map((f) => `expand[]=${encodeURIComponent(f)}`).join("&");
}

export async function fetchCatalogFromHF(): Promise<HFModelSummary[]> {
  const expand = buildExpandParams();
  const sliceUrls = [
    `${HF_API_BASE}/api/models?filter=text-generation&sort=downloads&direction=-1&limit=100&${expand}`,
    `${HF_API_BASE}/api/models?filter=gguf&sort=downloads&direction=-1&limit=100&${expand}`,
    `${HF_API_BASE}/api/models?filter=mlx&sort=downloads&direction=-1&limit=80&${expand}`,
    `${HF_API_BASE}/api/models?filter=image-text-to-text&sort=downloads&direction=-1&limit=50&${expand}`,
    `${HF_API_BASE}/api/models?filter=text-to-image&sort=downloads&direction=-1&limit=40&${expand}`,
    `${HF_API_BASE}/api/models?filter=text-generation&sort=trendingScore&direction=-1&limit=60&${expand}`,
    `${HF_API_BASE}/api/models?author=bartowski&sort=downloads&direction=-1&limit=80&${expand}`,
    `${HF_API_BASE}/api/models?author=unsloth&sort=downloads&direction=-1&limit=80&${expand}`,
    `${HF_API_BASE}/api/models?author=Qwen&sort=downloads&direction=-1&limit=60&${expand}`,
    `${HF_API_BASE}/api/models?author=meta-llama&sort=downloads&direction=-1&limit=40&${expand}`,
    `${HF_API_BASE}/api/models?author=mistralai&sort=downloads&direction=-1&limit=40&${expand}`,
    `${HF_API_BASE}/api/models?author=google&sort=downloads&direction=-1&limit=40&${expand}`,
    `${HF_API_BASE}/api/models?author=deepseek-ai&sort=downloads&direction=-1&limit=40&${expand}`,
  ];

  const responses = await Promise.allSettled(
    sliceUrls.map((url) => fetchJsonWithTimeout<RawHFApiModel[]>(url, 12000))
  );

  const byId = new Map<string, HFModelSummary>();

  for (const result of responses) {
    if (result.status !== "fulfilled" || !Array.isArray(result.value)) continue;
    for (const raw of result.value) {
      const normalized = normalizeHFSummary(raw);
      if (!normalized) continue;
      const existing = byId.get(normalized.id);
      if (!existing) {
        byId.set(normalized.id, normalized);
      } else {
        // Merge discovered formats cleanly
        const mergedFormats = Array.from(
          new Set([...existing.formats, ...normalized.formats])
        );
        byId.set(normalized.id, {
          ...existing,
          formats: mergedFormats,
          downloads: Math.max(existing.downloads, normalized.downloads),
        });
      }
    }
  }

  return Array.from(byId.values()).sort((a, b) => b.downloads - a.downloads);
}

export async function searchHFModels(query: string): Promise<HFModelSummary[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];

  const expand = buildExpandParams();
  const url = `${HF_API_BASE}/api/models?search=${encodeURIComponent(trimmed)}&limit=80&sort=downloads&direction=-1&${expand}`;
  const rawList = await fetchJsonWithTimeout<RawHFApiModel[]>(url, 10000);
  if (!Array.isArray(rawList)) return [];

  const seen = new Set<string>();
  const results: HFModelSummary[] = [];
  for (const raw of rawList) {
    const normalized = normalizeHFSummary(raw);
    if (normalized && !seen.has(normalized.id)) {
      seen.add(normalized.id);
      results.push(normalized);
    }
  }
  return results;
}

const GGUF_QUANT_PATTERNS: { regex: RegExp; mapped?: QuantizationLevel }[] = [
  { regex: /(?:^|[-_.])(Q2_K(?:_[A-Z])?)(?:[-_.]|$)/i, mapped: "Q2_K" },
  { regex: /(?:^|[-_.])(Q3_K_M|Q3_K_S|Q3_K_L)(?:[-_.]|$)/i, mapped: "Q3_K_M" },
  { regex: /(?:^|[-_.])(IQ4_XS|IQ4_NL)(?:[-_.]|$)/i, mapped: "IQ4_XS" },
  { regex: /(?:^|[-_.])(Q4_K_M|Q4_K_S|Q4_0)(?:[-_.]|$)/i, mapped: "Q4_K_M" },
  { regex: /(?:^|[-_.])(Q5_K_M|Q5_K_S|Q5_0)(?:[-_.]|$)/i, mapped: "Q5_K_M" },
  { regex: /(?:^|[-_.])(Q6_K)(?:[-_.]|$)/i, mapped: "Q6_K" },
  { regex: /(?:^|[-_.])(Q8_0)(?:[-_.]|$)/i, mapped: "Q8_0" },
  { regex: /(?:^|[-_.])(FP8|F8)(?:[-_.]|$)/i, mapped: "FP8" },
  { regex: /(?:^|[-_.])(BF16|F16|FP16)(?:[-_.]|$)/i, mapped: "FP16" },
];

function parseGgufQuantLabel(filename: string): {
  quantLabel: string;
  mappedQuant?: QuantizationLevel;
} {
  for (const { regex, mapped } of GGUF_QUANT_PATTERNS) {
    const m = filename.match(regex);
    if (m) {
      return { quantLabel: m[1].toUpperCase(), mappedQuant: mapped };
    }
  }
  return { quantLabel: "GGUF" };
}

function buildArchitectureFromConfig(
  summary: HFModelSummary,
  rawConfig: Record<string, unknown> | null,
  rawApi: RawHFApiModel | null
): ModelArchitectureConfig {
  const fallback = inferArchitectureFallback(summary);
  if (!rawConfig && !rawApi?.config && !rawApi?.gguf) {
    return fallback;
  }

  const root = rawConfig ?? rawApi?.config ?? {};
  const textCfg = (root.text_config as Record<string, unknown> | undefined) ?? root;
  const visionCfg = root.vision_config as Record<string, unknown> | undefined;

  const numLayers = Number(
    textCfg.num_hidden_layers ?? textCfg.n_layer ?? fallback.numLayers
  );
  const hiddenSize = Number(
    textCfg.hidden_size ?? textCfg.n_embd ?? textCfg.d_model ?? fallback.hiddenSize
  );
  const intermediateSize = Number(
    textCfg.intermediate_size ?? textCfg.ffn_dim ?? fallback.intermediateSize
  );
  const numAttentionHeads = Number(
    textCfg.num_attention_heads ?? textCfg.n_head ?? fallback.numAttentionHeads
  );
  const numKeyValueHeads = Number(
    textCfg.num_key_value_heads ?? numAttentionHeads ?? fallback.numKeyValueHeads
  );
  const vocabSize = Number(textCfg.vocab_size ?? fallback.vocabSize);
  const maxContextLength = Number(
    textCfg.max_position_embeddings ??
      rawApi?.gguf?.context_length ??
      fallback.maxContextLength
  );
  const ropeTheta = Number(textCfg.rope_theta ?? fallback.ropeTheta);
  const numExperts = Number(
    textCfg.num_local_experts ?? textCfg.n_routed_experts ?? fallback.numExperts ?? 0
  );
  const numActiveExperts = Number(
    textCfg.num_experts_per_tok ?? fallback.numActiveExperts ?? 0
  );

  const archArr = Array.isArray(root.architectures)
    ? (root.architectures as string[])
    : [];

  const isDiffusion =
    fallback.archetype === "diffusion" ||
    String(root._class_name ?? "").toLowerCase().includes("pipeline");

  const hasVisionEncoder = Boolean(visionCfg) || Boolean(fallback.hasVisionEncoder);
  const isMoE =
    numExperts > 1 ||
    summary.isMoE ||
    fallback.archetype === "moe-transformer";

  return {
    archetype: isDiffusion
      ? "diffusion"
      : isMoE
        ? "moe-transformer"
        : hasVisionEncoder
          ? "multimodal-vlm"
          : "dense-transformer",
    source: rawConfig ? "config.json" : rawApi?.gguf ? "gguf" : "estimated",
    modelType: String(textCfg.model_type ?? rawApi?.gguf?.architecture ?? fallback.modelType),
    architectureName:
      archArr[0] ??
      (typeof root._class_name === "string" ? root._class_name : undefined) ??
      rawApi?.gguf?.architecture ??
      fallback.architectureName,
    numLayers,
    hiddenSize,
    intermediateSize,
    numAttentionHeads,
    numKeyValueHeads,
    vocabSize,
    maxContextLength,
    ropeTheta,
    tieWordEmbeddings: Boolean(textCfg.tie_word_embeddings),
    numExperts: numExperts > 1 ? numExperts : undefined,
    numActiveExperts: numActiveExperts >= 1 ? numActiveExperts : undefined,
    hasVisionEncoder,
    visionLayers: visionCfg ? Number(visionCfg.num_hidden_layers ?? 24) : fallback.visionLayers,
    visionHiddenSize: visionCfg ? Number(visionCfg.hidden_size ?? 1152) : fallback.visionHiddenSize,
    rawConfig: rawConfig ?? (rawApi?.config as Record<string, unknown> | undefined),
  };
}

export async function fetchModelDetail(
  repoId: string,
  fallbackSummary?: HFModelSummary
): Promise<HFModelDetail> {
  if (!isValidRepoId(repoId)) {
    throw new Error(`Invalid repository ID: ${repoId}`);
  }

  const cached = detailCache.get(repoId);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.detail;
  }

  const expand = `${buildExpandParams()}&expand[]=siblings&expand[]=cardData`;
  const modelUrl = `${HF_API_BASE}/api/models/${repoId}?${expand}`;
  const treeUrl = `${HF_API_BASE}/api/models/${repoId}/tree/main`;
  const configUrl = `${HF_API_BASE}/${repoId}/resolve/main/config.json`;

  const [apiRes, treeRes, configRes] = await Promise.allSettled([
    fetchJsonWithTimeout<RawHFApiModel>(modelUrl, 11000),
    fetchJsonWithTimeout<RawTreeEntry[]>(treeUrl, 11000),
    fetchJsonWithTimeout<Record<string, unknown>>(configUrl, 9000),
  ]);

  const rawApi = apiRes.status === "fulfilled" ? apiRes.value : null;
  const rawTree = treeRes.status === "fulfilled" && Array.isArray(treeRes.value)
    ? treeRes.value
    : [];
  const rawConfig = configRes.status === "fulfilled" ? configRes.value : null;

  const summary =
    (rawApi ? normalizeHFSummary(rawApi) : null) ??
    fallbackSummary ?? {
      id: repoId,
      author: repoId.split("/")[0] ?? "huggingface",
      name: repoId.split("/").slice(1).join("/") || repoId,
      pipelineTag: "text-generation",
      libraryName: "transformers",
      formats: ["safetensors"],
      primaryFormat: "safetensors",
      parametersB: 7,
      activeParametersB: 7,
      isMoE: false,
      architectureHint: "Transformer",
      defaultContextLength: 32768,
      downloads: 0,
      likes: 0,
      gated: false,
      lastModified: new Date().toISOString(),
      tags: [],
      estimatedDiskGBAtFP16: 13.0,
      estimatedDiskGBAtQ4: 4.0,
    };

  // Extract weight files from tree/main (prefer LFS size!), fallback to siblings
  const weightFiles: ModelFileEntry[] = [];
  const sourceEntries: { path: string; sizeBytes: number }[] =
    rawTree.length > 0
      ? rawTree
          .filter((e) => e.type === "file")
          .map((e) => ({ path: e.path, sizeBytes: e.lfs?.size ?? e.size ?? 0 }))
      : (rawApi?.siblings ?? []).map((s) => ({
          path: s.rfilename,
          sizeBytes: s.lfs?.size ?? s.size ?? 0,
        }));

  const quantGroups = new Map<
    string,
    { mappedQuant?: QuantizationLevel; totalBytes: number; files: string[] }
  >();

  let totalSafetensorBytes = 0;
  let totalRepoWeightBytes = 0;

  for (const entry of sourceEntries) {
    const lower = entry.path.toLowerCase();
    if (lower.endsWith(".gguf")) {
      const { quantLabel, mappedQuant } = parseGgufQuantLabel(entry.path);
      weightFiles.push({
        path: entry.path,
        sizeBytes: entry.sizeBytes,
        format: "gguf",
        quantLabel,
        mappedQuant,
        isShard: /-0\d{4}-of-0\d{4}\.gguf$/i.test(entry.path),
      });
      const existing = quantGroups.get(quantLabel) ?? {
        mappedQuant,
        totalBytes: 0,
        files: [],
      };
      existing.totalBytes += entry.sizeBytes;
      existing.files.push(entry.path);
      quantGroups.set(quantLabel, existing);
      totalRepoWeightBytes += entry.sizeBytes;
    } else if (lower.endsWith(".safetensors")) {
      const isMlx = summary.formats.includes("mlx");
      weightFiles.push({
        path: entry.path,
        sizeBytes: entry.sizeBytes,
        format: isMlx ? "mlx" : "safetensors",
        quantLabel: isMlx ? "MLX" : "Safetensors",
        isShard: /-0\d{4}-of-0\d{4}\.safetensors$/i.test(entry.path),
      });
      totalSafetensorBytes += entry.sizeBytes;
      totalRepoWeightBytes += entry.sizeBytes;
    } else if (lower.endsWith(".bin") && lower.includes("pytorch_model")) {
      weightFiles.push({
        path: entry.path,
        sizeBytes: entry.sizeBytes,
        format: "safetensors",
        quantLabel: "PyTorch",
      });
      totalRepoWeightBytes += entry.sizeBytes;
    }
  }

  const quantVariants: QuantVariantEntry[] = Array.from(quantGroups.entries())
    .map(([quant, info]) => ({
      quant,
      mappedQuant: info.mappedQuant,
      totalBytes: info.totalBytes,
      files: info.files.sort(),
    }))
    .sort((a, b) => b.totalBytes - a.totalBytes);

  // Refine formats & sizes from actual files discovered
  const discoveredFormats = new Set<ModelFormat>(summary.formats);
  if (quantVariants.length > 0 && !discoveredFormats.has("gguf")) {
    discoveredFormats.add("gguf");
  }
  const formats = Array.from(discoveredFormats);

  const architecture = buildArchitectureFromConfig(summary, rawConfig, rawApi);
  const licenseRaw = rawApi?.cardData?.license;
  const license = Array.isArray(licenseRaw) ? licenseRaw[0] : licenseRaw;

  const detail: HFModelDetail = {
    ...summary,
    formats,
    primaryFormat: formats[0],
    defaultContextLength: architecture.maxContextLength,
    exactBaseBytes: totalSafetensorBytes > 0 ? totalSafetensorBytes : summary.exactBaseBytes,
    files: weightFiles.sort((a, b) => b.sizeBytes - a.sizeBytes).slice(0, 80),
    quantVariants,
    totalRepoWeightBytes,
    architecture,
    license,
  };

  setCachedDetail(repoId, detail);
  return detail;
}
