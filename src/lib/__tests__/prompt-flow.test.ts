import { describe, expect, it } from "bun:test";
import type { HFModelSummary } from "@/types/inference";
import { buildArchitectureBlueprint } from "@/lib/architecture-graph";
import { DEFAULT_INFERENCE_CONFIG } from "@/lib/inference-math";
import {
  buildCandidatesForWord,
  buildPromptFlowSimulation,
  tokenizePromptText,
} from "@/lib/prompt-flow";

const SAMPLE_LLM: HFModelSummary = {
  id: "Qwen/Qwen2.5-7B-Instruct",
  author: "Qwen",
  name: "Qwen2.5-7B-Instruct",
  pipelineTag: "text-generation",
  libraryName: "transformers",
  formats: ["safetensors"],
  primaryFormat: "safetensors",
  parametersB: 7.6,
  activeParametersB: 7.6,
  isMoE: false,
  architectureHint: "Qwen2ForCausalLM",
  defaultContextLength: 32768,
  downloads: 1000000,
  likes: 3000,
  gated: false,
  lastModified: "2026-01-01T00:00:00Z",
  tags: ["text-generation"],
  estimatedDiskGBAtFP16: 15.2,
  estimatedDiskGBAtQ4: 4.6,
};

const SAMPLE_MOE: HFModelSummary = {
  ...SAMPLE_LLM,
  id: "deepseek-ai/DeepSeek-V3",
  name: "DeepSeek-V3",
  isMoE: true,
  parametersB: 671,
  activeParametersB: 37,
  architectureHint: "DeepseekV3ForCausalLM",
};

const SAMPLE_DIFFUSION: HFModelSummary = {
  ...SAMPLE_LLM,
  id: "black-forest-labs/FLUX.1-dev",
  name: "FLUX.1-dev",
  pipelineTag: "text-to-image",
  formats: ["diffusers"],
  primaryFormat: "diffusers",
  parametersB: 12,
  activeParametersB: 12,
  architectureHint: "FluxPipeline",
};

describe("prompt-flow tokenization and step-by-step simulation", () => {
  it("tokenizes sentences, punctuation, and long words deterministically within vocab bounds", () => {
    const tokens = tokenizePromptText("How does local architecture scale!", 32000);
    expect(tokens.length).toBeGreaterThanOrEqual(6);
    expect(tokens.some((t) => t.text === "architect")).toBe(true);
    for (const tok of tokens) {
      expect(tok.tokenId).toBeGreaterThanOrEqual(100);
      expect(tok.tokenId).toBeLessThan(32000);
      expect(tok.embeddingPreview.length).toBe(4);
    }

    const fallbackTokens = tokenizePromptText("   ", 50000);
    expect(fallbackTokens.some((t) => t.text.toLowerCase() === "sky")).toBe(true);
  });

  it("builds prefill and autoregressive decode steps with valid 3D stage references and Softmax sums", () => {
    const bp = buildArchitectureBlueprint(SAMPLE_LLM, DEFAULT_INFERENCE_CONFIG);
    const stageIds = new Set(bp.stages.map((s) => s.id));

    const sim = buildPromptFlowSimulation(
      "How does local AI run on my Mac?",
      SAMPLE_LLM,
      bp,
      DEFAULT_INFERENCE_CONFIG
    );

    expect(sim.steps.length).toBeGreaterThan(8);
    expect(sim.steps.some((s) => s.phase === "prefill")).toBe(true);
    expect(sim.steps.some((s) => s.phase === "decode" && s.feedbackLoopActive)).toBe(
      true
    );

    for (const step of sim.steps) {
      expect(stageIds.has(step.stageId)).toBe(true);
      expect(stageIds.has(step.fromStageId)).toBe(true);
      if (step.candidates) {
        const totalPct = step.candidates.reduce((s, c) => s + c.probPct, 0);
        expect(totalPct).toBe(100);
        expect(step.candidates.some((c) => c.chosen)).toBe(true);
      }
      if (step.attentionLinks) {
        expect(step.attentionLinks.length).toBeGreaterThan(0);
      }
    }

    const lastStep = sim.steps[sim.steps.length - 1];
    expect(lastStep.generatedSoFar.length).toBe(sim.answerWords.length);
  });

  it("supports MoE and diffusion blueprints gracefully without missing stages", () => {
    const moeBp = buildArchitectureBlueprint(SAMPLE_MOE, DEFAULT_INFERENCE_CONFIG);
    const moeSim = buildPromptFlowSimulation(
      "Explain attention routing",
      SAMPLE_MOE,
      moeBp,
      DEFAULT_INFERENCE_CONFIG
    );
    expect(moeSim.steps.length).toBeGreaterThan(4);

    const diffBp = buildArchitectureBlueprint(
      SAMPLE_DIFFUSION,
      DEFAULT_INFERENCE_CONFIG
    );
    const diffSim = buildPromptFlowSimulation(
      "A ceramic cube studio",
      SAMPLE_DIFFUSION,
      diffBp,
      DEFAULT_INFERENCE_CONFIG
    );
    expect(diffSim.steps.length).toBeGreaterThanOrEqual(diffBp.stages.length - 2);
    expect(buildCandidatesForWord("test", 3).reduce((a, b) => a + b.probPct, 0)).toBe(
      100
    );
  });
});
