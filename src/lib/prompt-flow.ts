import type {
  HFModelDetail,
  HFModelSummary,
  InferenceConfig,
} from "@/types/inference";
import type {
  ArchitectureBlueprint,
  ArchitectureStageNode,
} from "@/lib/architecture-graph";

export interface PromptToken {
  index: number;
  text: string;
  tokenId: number;
  embeddingPreview: [number, number, number, number];
}

export interface TokenCandidate {
  token: string;
  probPct: number;
  chosen: boolean;
}

export interface AttentionLink {
  fromWord: string;
  toWord: string;
  weightPct: number;
}

export interface PromptFlowStep {
  stepIndex: number;
  totalSteps: number;
  phase: "prefill" | "decode";
  stageId: string;
  fromStageId: string;
  stageShortLabel: string;
  activeWords: string[];
  generatedSoFar: string[];
  kvCachedWords: string[];
  title: string;
  explanation: string;
  tensorBadge: string;
  kvWriteActive: boolean;
  feedbackLoopActive: boolean;
  candidates?: TokenCandidate[];
  attentionLinks?: AttentionLink[];
}

export interface PromptFlowSimulation {
  promptText: string;
  promptTokens: PromptToken[];
  answerWords: string[];
  fullAnswerText: string;
  steps: PromptFlowStep[];
}

const DEFAULT_PROMPT = "Why is the sky blue?";

function fnv1aHash(str: string): number {
  let hash = 2166136261;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function tokenizePromptText(
  text: string,
  vocabSize = 128256
): PromptToken[] {
  const cleaned = text.trim() || DEFAULT_PROMPT;
  const rawParts = cleaned.match(/[A-Za-z0-9_'-]+|[.,!?;:()]/g) ?? [cleaned];
  const pieces: string[] = [];

  for (const part of rawParts) {
    if (part.length > 10 && /^[A-Za-z]+$/.test(part)) {
      const splitAt = Math.ceil(part.length * 0.68);
      pieces.push(part.slice(0, splitAt));
      pieces.push(part.slice(splitAt));
    } else {
      pieces.push(part);
    }
    if (pieces.length >= 24) break;
  }

  const maxVocab = Math.max(200, vocabSize - 1);
  return pieces.map((piece, index) => {
    const h = fnv1aHash(`${piece}:${index}`);
    const tokenId = 100 + (h % (maxVocab - 100));
    const vec: [number, number, number, number] = [
      Number((((((h >>> 0) & 0xff) / 255) * 2 - 1)).toFixed(2)),
      Number((((((h >>> 8) & 0xff) / 255) * 2 - 1)).toFixed(2)),
      Number((((((h >>> 16) & 0xff) / 255) * 2 - 1)).toFixed(2)),
      Number((((((h >>> 24) & 0xff) / 255) * 2 - 1)).toFixed(2)),
    ];
    return {
      index,
      text: piece,
      tokenId,
      embeddingPreview: vec,
    };
  });
}

const ALT_CANDIDATE_POOLS: string[][] = [
  ["likely", "often", "core", "clear"],
  ["energy", "signal", "layer", "tensor"],
  ["matrix", "context", "memory", "vector"],
  ["fast", "local", "parallel", "cached"],
  ["because", "through", "across", "during"],
];

export function buildCandidatesForWord(
  chosenWord: string,
  seed: number
): TokenCandidate[] {
  const pool = ALT_CANDIDATE_POOLS[Math.abs(seed) % ALT_CANDIDATE_POOLS.length];
  const distinctAlts = pool.filter(
    (w) => w.toLowerCase() !== chosenWord.toLowerCase()
  );
  const alt1 = distinctAlts[0] ?? "also";
  const alt2 = distinctAlts[1] ?? "then";
  const alt3 = distinctAlts[2] ?? "here";

  const pChosen = 64 + (Math.abs(seed) % 18); // 64..81
  const rem = 100 - pChosen;
  const p1 = Math.max(1, Math.round(rem * 0.52));
  const p2 = Math.max(1, Math.round(rem * 0.31));
  const p3 = Math.max(1, rem - p1 - p2);

  return [
    { token: chosenWord, probPct: pChosen, chosen: true },
    { token: alt1, probPct: p1, chosen: false },
    { token: alt2, probPct: p2, chosen: false },
    { token: alt3, probPct: p3, chosen: false },
  ];
}

export function synthesizeAnswerWords(
  promptText: string,
  model: HFModelSummary | HFModelDetail
): string[] {
  const lower = (promptText.trim() || DEFAULT_PROMPT).toLowerCase();

  if (/\b(hello|hi|hey|greetings|morning)\b/.test(lower)) {
    return ["Hello!", "Local", "inference", "runs", "weights", "inside", "memory."];
  }
  if (/\b(sky|blue|sun|light|scatter|rayleigh)\b/.test(lower)) {
    return ["Sunlight", "scatters", "short", "blue", "waves", "across", "air."];
  }
  if (/\b(mac|apple|silicon|gpu|vram|metal|memory|hardware|local|run)\b/.test(lower)) {
    return ["Quantized", "weights", "stream", "via", "unified", "memory", "fast."];
  }
  if (/\b(code|python|typescript|javascript|rust|sort|function|bug)\b/.test(lower)) {
    return ["Split", "arrays", "recursively,", "sort", "halves,", "then", "merge."];
  }
  if (/\b(transformer|attention|ai|llm|kv|moe|model|token|neural)\b/.test(lower)) {
    return ["Attention", "links", "tokens", "while", "FFN", "refines", "meanings."];
  }
  if (/\d+/.test(lower)) {
    return ["Computed", "via", "weighted", "logits", "across", "active", "layers."];
  }

  // Contextual synthesis anchoring onto user's content keywords
  const stopwords = new Set([
    "what",
    "why",
    "how",
    "when",
    "where",
    "does",
    "this",
    "that",
    "with",
    "from",
    "into",
    "about",
    "your",
    "they",
    "will",
    "have",
  ]);
  const keywords = (promptText.match(/[A-Za-z0-9]+/g) ?? [])
    .filter((w) => w.length >= 3 && !stopwords.has(w.toLowerCase()))
    .slice(0, 2);

  if (keywords.length >= 2) {
    return [
      keywords[0],
      "aligns",
      keywords[1],
      "through",
      "cached",
      "attention",
      "layers.",
    ];
  }
  if (keywords.length === 1) {
    return [
      model.name.split("-")[0] || "Model",
      "maps",
      keywords[0],
      "into",
      "cached",
      "next",
      "tokens.",
    ];
  }
  return ["Context", "flows", "through", "attention", "into", "sampled", "tokens."];
}

function buildAttentionLinks(
  promptWords: string[],
  focusWord: string,
  stepSeed: number
): AttentionLink[] {
  const sample = promptWords.slice(0, 4);
  if (sample.length === 0) return [];
  const raw = sample.map((w, idx) => ({
    word: w,
    score: 20 + ((fnv1aHash(`${focusWord}:${w}:${stepSeed}:${idx}`) % 55)),
  }));
  const sum = Math.max(1, raw.reduce((s, r) => s + r.score, 0));
  let running = 0;
  return raw.map((r, idx) => {
    const weight =
      idx === raw.length - 1
        ? Math.max(1, 100 - running)
        : Math.max(1, Math.round((r.score / sum) * 100));
    running += weight;
    return {
      fromWord: focusWord,
      toWord: r.word,
      weightPct: weight,
    };
  });
}

function describePrefillStage(
  stage: ArchitectureStageNode,
  tokenStrings: string[],
  arch: ArchitectureBlueprint["architecture"]
): { title: string; explanation: string; kvWrite: boolean } {
  const preview = tokenStrings.slice(0, 5).join(" · ");
  switch (stage.category) {
    case "input":
      return {
        title: "Tokenize User Sentence into Integer IDs",
        explanation: `Your text splits into ${tokenStrings.length} subword tokens [${preview}], indexed in the ${arch.vocabSize.toLocaleString()}-token vocabulary.`,
        kvWrite: false,
      };
    case "embedding":
      return {
        title: "Lookup & Positional Encoding (Embeddings)",
        explanation: `Each word vector pulls its ${arch.hiddenSize}-dimensional embedding and receives RoPE positional angles so word order is preserved.`,
        kvWrite: false,
      };
    case "attention":
      return {
        title: `${stage.label}: Self-Attention & KV Write`,
        explanation: `Every prompt word compares Query × Key vectors against earlier words (${arch.numAttentionHeads}Q / ${arch.numKeyValueHeads}KV) and caches K/V states for reuse.`,
        kvWrite: true,
      };
    case "ffn":
    case "moe-router":
    case "moe-expert":
      return {
        title: `${stage.label}: FFN / Expert Transformation`,
        explanation:
          arch.numExperts && arch.numExperts > 1
            ? `Router sends each token to top-${arch.numActiveExperts ?? 2} of ${arch.numExperts} experts (${arch.intermediateSize}-wide FFN) to enrich factual & syntactic features.`
            : `Up-projects hidden vectors to ${arch.intermediateSize}d with SwiGLU gating and projects refined meaning back into the residual stream.`,
        kvWrite: false,
      };
    case "kv-cache":
      return {
        title: "Store Key/Value Prompt Context in Cache",
        explanation: `All ${tokenStrings.length} prompt tokens now reside in GPU KV Memory so upcoming answer tokens read them in O(1) without recomputing the prompt.`,
        kvWrite: true,
      };
    case "norm":
    case "head":
    case "output":
      return {
        title: "Final RMSNorm & LM Head Logit Projection",
        explanation: `Normalizes the last token hidden vector and multiplies by [${arch.hiddenSize} × ${arch.vocabSize.toLocaleString()}] vocabulary matrix to score every candidate next word.`,
        kvWrite: false,
      };
    default:
      return {
        title: stage.label,
        explanation: stage.description,
        kvWrite: false,
      };
  }
}

export function buildPromptFlowSimulation(
  promptText: string,
  model: HFModelSummary | HFModelDetail,
  blueprint: ArchitectureBlueprint,
  config: InferenceConfig
): PromptFlowSimulation {
  const normalizedPrompt = promptText.trim() || DEFAULT_PROMPT;
  const arch = blueprint.architecture;
  const promptTokens = tokenizePromptText(normalizedPrompt, arch.vocabSize);
  const promptWords = promptTokens.map((t) => t.text);
  const answerWords = synthesizeAnswerWords(normalizedPrompt, model).slice(0, 7);

  const spineStages = [...blueprint.stages]
    .filter((s) => Math.abs(s.position[2]) < 1.2)
    .sort((a, b) => a.position[0] - b.position[0]);

  const flowStages = spineStages.length > 0 ? spineStages : [...blueprint.stages];
  const firstStage = flowStages[0];
  const lastStage = flowStages[flowStages.length - 1];
  const firstAttnStage =
    flowStages.find((s) => s.category === "attention") ??
    flowStages[Math.min(1, flowStages.length - 1)];
  const midFfnStage =
    flowStages.find(
      (s) =>
        s.category === "ffn" ||
        s.category === "moe-expert" ||
        s.category === "moe-router"
    ) ?? firstAttnStage;

  const stepsWithoutTotals: Omit<PromptFlowStep, "stepIndex" | "totalSteps">[] = [];

  let kvCachedWords: string[] = [];
  const generatedSoFar: string[] = [];

  // Phase 1: Prefill walk across main pipeline stages (sampling word #1 at the final stage)
  const prefillSample = promptWords.slice(0, 4);
  for (let i = 0; i < flowStages.length; i++) {
    const st = flowStages[i];
    const prevSt = i > 0 ? flowStages[i - 1] : firstStage;
    const isFinalPrefill = i === flowStages.length - 1;
    const info = describePrefillStage(st, promptWords, arch);

    if (info.kvWrite || st.category === "attention") {
      kvCachedWords = [...promptWords];
    }

    if (isFinalPrefill && answerWords.length > 0) {
      const firstAns = answerWords[0];
      generatedSoFar.push(firstAns);
      stepsWithoutTotals.push({
        phase: "prefill",
        stageId: st.id,
        fromStageId: prevSt.id,
        stageShortLabel: st.shortLabel,
        activeWords: [firstAns],
        generatedSoFar: [...generatedSoFar],
        kvCachedWords: [...kvCachedWords],
        title: `Sample Answer Word #1: "${firstAns}"`,
        explanation: `Softmax selects "${firstAns}" from top vocabulary probabilities after full-prompt prefill at ${config.quantization}.`,
        tensorBadge: `[B=${config.batchSize}, S=${promptTokens.length} → 1, V=${arch.vocabSize.toLocaleString()}]`,
        kvWriteActive: false,
        feedbackLoopActive: false,
        candidates: buildCandidatesForWord(firstAns, 11),
      });
    } else {
      const isAttn = st.category === "attention";
      stepsWithoutTotals.push({
        phase: "prefill",
        stageId: st.id,
        fromStageId: prevSt.id,
        stageShortLabel: st.shortLabel,
        activeWords: prefillSample,
        generatedSoFar: [...generatedSoFar],
        kvCachedWords: [...kvCachedWords],
        title: info.title,
        explanation: info.explanation,
        tensorBadge: `[B=${config.batchSize}, S=${promptTokens.length}, d=${arch.hiddenSize}]`,
        kvWriteActive: info.kvWrite || isAttn,
        feedbackLoopActive: false,
        attentionLinks: isAttn
          ? buildAttentionLinks(
              promptWords,
              promptWords[promptWords.length - 1] ?? "context",
              i
            )
          : undefined,
      });
    }
  }

  // Phase 2: Autoregressive Decode feedback loop generating words #2 .. #N one by one
  if (blueprint.archetype !== "diffusion") {
    for (let wIdx = 1; wIdx < answerWords.length; wIdx++) {
      const incomingToken = answerWords[wIdx - 1];
      const predictedToken = answerWords[wIdx];

      kvCachedWords = [...kvCachedWords, incomingToken];

      // Feedback + KV read/append step
      stepsWithoutTotals.push({
        phase: "decode",
        stageId: firstAttnStage.id,
        fromStageId: lastStage.id,
        stageShortLabel: firstAttnStage.shortLabel,
        activeWords: [incomingToken],
        generatedSoFar: [...generatedSoFar],
        kvCachedWords: [...kvCachedWords],
        title: `Decode Step ${wIdx + 1}: Feed "${incomingToken}" & Attend KV Cache`,
        explanation: `Previously generated "${incomingToken}" loops back as a single token [S=1], appends its K/V pair (${kvCachedWords.length} tokens cached), and attends to all prior words.`,
        tensorBadge: `[B=${config.batchSize}, S=1, kv=${kvCachedWords.length}, d=${arch.hiddenSize}]`,
        kvWriteActive: true,
        feedbackLoopActive: true,
        attentionLinks: buildAttentionLinks(promptWords, incomingToken, wIdx * 17),
      });

      // FFN + LM Head sample next word
      generatedSoFar.push(predictedToken);
      stepsWithoutTotals.push({
        phase: "decode",
        stageId: lastStage.id,
        fromStageId: midFfnStage.id,
        stageShortLabel: lastStage.shortLabel,
        activeWords: [predictedToken],
        generatedSoFar: [...generatedSoFar],
        kvCachedWords: [...kvCachedWords],
        title: `Emit Answer Word #${wIdx + 1}: "${predictedToken}"`,
        explanation: `FFN + LM Head transforms the single-token vector into logits and samples "${predictedToken}" for the streaming output.`,
        tensorBadge: `[B=${config.batchSize}, S=1, V=${arch.vocabSize.toLocaleString()}]`,
        kvWriteActive: false,
        feedbackLoopActive: false,
        candidates: buildCandidatesForWord(predictedToken, (wIdx + 1) * 31),
      });
    }
  }

  const totalSteps = stepsWithoutTotals.length;
  const steps: PromptFlowStep[] = stepsWithoutTotals.map((s, idx) => ({
    ...s,
    stepIndex: idx,
    totalSteps,
  }));

  return {
    promptText: normalizedPrompt,
    promptTokens,
    answerWords,
    fullAnswerText: answerWords.join(" "),
    steps,
  };
}
