import { describe, expect, it } from "bun:test";
import type { HFModelSummary } from "@/types/inference";
import {
  getDeviceById,
  getHardwareGroupedByVendor,
  HARDWARE_CATALOG,
} from "@/lib/hardware-catalog";
import {
  calculateInference,
  DEFAULT_INFERENCE_CONFIG,
  filterSupportedModelsForDevice,
  generateLocalAIYaml,
} from "@/lib/inference-math";

function makeModel(overrides: Partial<HFModelSummary> = {}): HFModelSummary {
  return {
    id: "Qwen/Qwen2.5-7B-Instruct",
    author: "Qwen",
    name: "Qwen2.5-7B-Instruct",
    pipelineTag: "text-generation",
    libraryName: "transformers",
    formats: ["safetensors", "gguf"],
    primaryFormat: "safetensors",
    parametersB: 7.6,
    activeParametersB: 7.6,
    isMoE: false,
    architectureHint: "Qwen2ForCausalLM",
    defaultContextLength: 32768,
    downloads: 1200000,
    likes: 4200,
    gated: false,
    lastModified: "2026-01-01T00:00:00Z",
    tags: ["text-generation"],
    estimatedDiskGBAtFP16: 15.2,
    estimatedDiskGBAtQ4: 4.6,
    ...overrides,
  };
}

describe("hardware-catalog & inference-math", () => {
  it("exposes at least 70 market accelerators grouped by vendor including Apple Silicon", () => {
    expect(HARDWARE_CATALOG.length).toBeGreaterThanOrEqual(70);

    const groups = getHardwareGroupedByVendor();
    expect(groups.length).toBeGreaterThanOrEqual(5);
    const totalGrouped = groups.reduce((sum, g) => sum + g.devices.length, 0);
    expect(totalGrouped).toBe(HARDWARE_CATALOG.length);

    expect(HARDWARE_CATALOG.some((d) => d.id === "apple-m4-max-64")).toBe(true);
    expect(HARDWARE_CATALOG.some((d) => d.id.startsWith("apple-m5"))).toBe(true);
    expect(HARDWARE_CATALOG.some((d) => d.id.includes("5090"))).toBe(true);
    expect(HARDWARE_CATALOG.some((d) => d.id.includes("mi325x"))).toBe(true);
  });

  it("fits a 7B Q4_K_M model comfortably on M4 Max 64GB and scales KV with context", () => {
    const dev = getDeviceById("apple-m4-max-64");
    const small = makeModel();

    const est8k = calculateInference(small, dev, {
      ...DEFAULT_INFERENCE_CONFIG,
      contextLength: 8192,
    });
    const est32k = calculateInference(small, dev, {
      ...DEFAULT_INFERENCE_CONFIG,
      contextLength: 32768,
    });

    expect(est8k.fits).toBe(true);
    expect(est8k.supported).toBe(true);
    expect(est32k.kvCacheGB).toBeGreaterThan(est8k.kvCacheGB * 3);
    expect(est8k.estimatedDecodeTps).toBeGreaterThan(10);
  });

  it("strictly filters out OOM models for a given hardware accelerator", () => {
    const rtx8gb = HARDWARE_CATALOG.find((d) => d.vramGB <= 8)!;
    const small7b = makeModel({ id: "small/7b", parametersB: 7, formats: ["gguf"], primaryFormat: "gguf" });
    const huge70b = makeModel({ id: "huge/70b", parametersB: 70, formats: ["gguf"], primaryFormat: "gguf" });

    const supported = filterSupportedModelsForDevice([small7b, huge70b], rtx8gb, {
      ...DEFAULT_INFERENCE_CONFIG,
      quantization: "Q4_K_M",
      contextLength: 8192,
    });

    expect(supported.some((x) => x.model.id === "small/7b")).toBe(true);
    expect(supported.some((x) => x.model.id === "huge/70b")).toBe(false);
  });

  it("generates valid LocalAI model.yaml snippet", () => {
    const dev = getDeviceById("apple-m4-max-64");
    const m = makeModel();
    const est = calculateInference(m, dev, DEFAULT_INFERENCE_CONFIG);
    const yaml = generateLocalAIYaml(m, DEFAULT_INFERENCE_CONFIG, est);

    expect(yaml).toContain("name: qwen2.5-7b-instruct");
    expect(yaml).toContain("huggingface://Qwen/Qwen2.5-7B-Instruct");
    expect(yaml).toContain("context_size: 8192");
  });
});
