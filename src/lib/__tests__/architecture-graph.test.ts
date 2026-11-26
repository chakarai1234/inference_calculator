import { describe, expect, it } from "bun:test";
import type { HFModelSummary } from "@/types/inference";
import { DEFAULT_INFERENCE_CONFIG } from "@/lib/inference-math";
import {
  buildArchitectureBlueprint,
  buildSingleLayerBreakdown,
} from "@/lib/architecture-graph";

const DENSE_MODEL: HFModelSummary = {
  id: "meta-llama/Llama-3.1-8B-Instruct",
  author: "meta-llama",
  name: "Llama-3.1-8B-Instruct",
  pipelineTag: "text-generation",
  libraryName: "transformers",
  formats: ["safetensors", "gguf"],
  primaryFormat: "safetensors",
  parametersB: 8.0,
  activeParametersB: 8.0,
  isMoE: false,
  architectureHint: "LlamaForCausalLM",
  defaultContextLength: 131072,
  downloads: 5000000,
  likes: 8000,
  gated: false,
  lastModified: "2026-01-01T00:00:00Z",
  tags: ["text-generation"],
  estimatedDiskGBAtFP16: 16.0,
  estimatedDiskGBAtQ4: 4.85,
  embeddedConfig: {
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 14336,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    vocabSize: 128256,
    maxContextLength: 131072,
  },
};

const MOE_MODEL: HFModelSummary = {
  id: "mistralai/Mixtral-8x7B-Instruct-v0.1",
  author: "mistralai",
  name: "Mixtral-8x7B-Instruct-v0.1",
  pipelineTag: "text-generation",
  libraryName: "transformers",
  formats: ["safetensors", "gguf"],
  primaryFormat: "safetensors",
  parametersB: 46.7,
  activeParametersB: 12.9,
  isMoE: true,
  architectureHint: "MixtralForCausalLM",
  defaultContextLength: 32768,
  downloads: 2000000,
  likes: 4000,
  gated: false,
  lastModified: "2026-01-01T00:00:00Z",
  tags: ["text-generation", "moe"],
  estimatedDiskGBAtFP16: 93.4,
  estimatedDiskGBAtQ4: 28.3,
  embeddedConfig: {
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 14336,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    numExperts: 8,
    numActiveExperts: 2,
    vocabSize: 32000,
    maxContextLength: 32768,
  },
};

describe("architecture-graph & single-layer breakdown", () => {
  it("annotates transformer stage blocks with clickable layerStart and layerEnd ranges", () => {
    const bp = buildArchitectureBlueprint(DENSE_MODEL, DEFAULT_INFERENCE_CONFIG);
    const layeredStages = bp.stages.filter((s) => s.layerStart !== undefined);

    expect(layeredStages.length).toBeGreaterThanOrEqual(4);
    expect(layeredStages[0].layerStart).toBe(1);
    const lastStage = layeredStages[layeredStages.length - 1];
    expect(lastStage.layerEnd).toBe(32);
  });

  it("computes single-layer forward pass breakdown with valid memory, KV, and FLOPs", () => {
    const denseLayer = buildSingleLayerBreakdown(DENSE_MODEL, DEFAULT_INFERENCE_CONFIG, 12);
    expect(denseLayer.layerNumber).toBe(12);
    expect(denseLayer.totalLayers).toBe(32);
    expect(denseLayer.totalLayerParams).toBeGreaterThan(0);
    expect(denseLayer.totalLayerMemoryMB).toBeGreaterThan(0);
    expect(denseLayer.kvCacheThisLayerMB).toBeGreaterThan(0);
    expect(denseLayer.layerFlopsPerTokenG).toBeGreaterThan(0);
    expect(denseLayer.attnSharePct + denseLayer.ffnSharePct).toBeCloseTo(100, 0);
    expect(denseLayer.subSteps.length).toBeGreaterThanOrEqual(5);

    const moeLayer = buildSingleLayerBreakdown(MOE_MODEL, DEFAULT_INFERENCE_CONFIG, 1);
    expect(moeLayer.activeLayerParams).toBeLessThan(moeLayer.totalLayerParams);
    expect(moeLayer.activeLayerMemoryMB).toBeLessThan(moeLayer.totalLayerMemoryMB);
    expect(moeLayer.subSteps.some((s) => s.name.toLowerCase().includes("moe"))).toBe(true);
  });
});
