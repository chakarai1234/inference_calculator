import { describe, expect, it } from "bun:test";
import {
  buildContextScaleSweep,
  buildParameterScaleSweep,
  computeFormulaLab,
  DEFAULT_FORMULA_LAB_INPUT,
} from "@/lib/formula-lab";

describe("formula-lab full-range calculator", () => {
  it("computes finite monotonic weights across sub-billion to multi-trillion scales", () => {
    const scales = [0.5, 8, 70, 405, 671, 1000, 2000, 5000];
    let prevWeights = 0;

    for (const totalParamsB of scales) {
      const res = computeFormulaLab({
        ...DEFAULT_FORMULA_LAB_INPUT,
        totalParamsB,
        activeParamsB: totalParamsB,
      });

      expect(Number.isFinite(res.weightsGB)).toBe(true);
      expect(Number.isFinite(res.kvCacheGB)).toBe(true);
      expect(Number.isFinite(res.totalRequiredGB)).toBe(true);
      expect(Number.isFinite(res.estimatedDecodeTps)).toBe(true);
      expect(res.weightsGB).toBeGreaterThan(prevWeights);
      expect(res.liveEquations.length).toBeGreaterThanOrEqual(4);
      prevWeights = res.weightsGB;
    }
  });

  it("scales quantization accurately across FP32, FP16, Q4_K_M, and BitNet 1.58b", () => {
    const fp16 = computeFormulaLab({
      ...DEFAULT_FORMULA_LAB_INPUT,
      totalParamsB: 70,
      bitsPerWeight: 16,
    });
    const fp32 = computeFormulaLab({
      ...DEFAULT_FORMULA_LAB_INPUT,
      totalParamsB: 70,
      bitsPerWeight: 32,
    });
    const q4 = computeFormulaLab({
      ...DEFAULT_FORMULA_LAB_INPUT,
      totalParamsB: 70,
      bitsPerWeight: 4.85,
    });
    const bitnet = computeFormulaLab({
      ...DEFAULT_FORMULA_LAB_INPUT,
      totalParamsB: 70,
      bitsPerWeight: 1.58,
    });

    expect(fp32.weightsGB).toBeCloseTo(fp16.weightsGB * 2, 2);
    expect(q4.weightsGB).toBeCloseTo(fp16.weightsGB * (4.85 / 16), 2);
    expect(bitnet.weightsGB).toBeLessThan(q4.weightsGB);
  });

  it("orders attention KV cache memory strictly MHA > GQA-4 > GQA-8 > MQA and DeepSeek MLA lowest", () => {
    const base = {
      ...DEFAULT_FORMULA_LAB_INPUT,
      totalParamsB: 70,
      contextLength: 131072,
    };

    const mha = computeFormulaLab({ ...base, attentionMode: "mha" });
    const gqa4 = computeFormulaLab({ ...base, attentionMode: "gqa-4" });
    const gqa8 = computeFormulaLab({ ...base, attentionMode: "gqa-8" });
    const mqa = computeFormulaLab({ ...base, attentionMode: "mqa" });
    const mla = computeFormulaLab({ ...base, attentionMode: "mla" });

    expect(mha.kvCacheGB).toBeGreaterThan(gqa4.kvCacheGB);
    expect(gqa4.kvCacheGB).toBeGreaterThan(gqa8.kvCacheGB);
    expect(gqa8.kvCacheGB).toBeGreaterThan(mqa.kvCacheGB);
    expect(mla.kvCacheGB).toBeLessThan(gqa8.kvCacheGB);
  });

  it("models sparse MoE weights from totalParamsB and decode speed from activeParamsB", () => {
    const dense671 = computeFormulaLab({
      ...DEFAULT_FORMULA_LAB_INPUT,
      totalParamsB: 671,
      isMoE: false,
      activeParamsB: 671,
      vramPerGpuGB: 512,
      bandwidthGBs: 819,
    });
    const moe671 = computeFormulaLab({
      ...DEFAULT_FORMULA_LAB_INPUT,
      totalParamsB: 671,
      isMoE: true,
      activeParamsB: 37,
      vramPerGpuGB: 512,
      bandwidthGBs: 819,
    });

    expect(moe671.weightsGB).toBeCloseTo(dense671.weightsGB, 4);
    expect(moe671.estimatedDecodeTps).toBeGreaterThan(dense671.estimatedDecodeTps * 5);
  });

  it("computes minGpusNeeded and maxContextThatFits across multi-GPU and batch scaling", () => {
    const cluster1T = computeFormulaLab({
      ...DEFAULT_FORMULA_LAB_INPUT,
      totalParamsB: 1000,
      bitsPerWeight: 8.2,
      contextLength: 131072,
      vramPerGpuGB: 80,
      gpuCount: 8,
      usableMemoryRatio: 0.95,
    });
    expect(cluster1T.fits).toBe(false);
    expect(cluster1T.minGpusNeeded).toBeGreaterThanOrEqual(12);

    const b1 = computeFormulaLab({
      ...DEFAULT_FORMULA_LAB_INPUT,
      totalParamsB: 8,
      vramPerGpuGB: 24,
      batchSize: 1,
    });
    const b2 = computeFormulaLab({
      ...DEFAULT_FORMULA_LAB_INPUT,
      totalParamsB: 8,
      vramPerGpuGB: 24,
      batchSize: 2,
    });
    expect(b2.maxContextThatFits).toBeLessThan(b1.maxContextThatFits);
  });

  it("honors manual architecture overrides directly", () => {
    const base = computeFormulaLab({
      ...DEFAULT_FORMULA_LAB_INPUT,
      totalParamsB: 8,
      contextLength: 32768,
    });
    const doubledLayers = computeFormulaLab({
      ...DEFAULT_FORMULA_LAB_INPUT,
      totalParamsB: 8,
      contextLength: 32768,
      overrideLayers: base.architecture.layers * 2,
      overrideKvHeads: base.architecture.kvHeads,
      overrideHeadDim: base.architecture.headDim,
    });

    expect(doubledLayers.isAutoArch).toBe(false);
    expect(doubledLayers.kvCacheGB).toBeCloseTo(base.kvCacheGB * 2, 3);
  });

  it("generates full parameter (0.5B..2T) and context (4K..2M) scaling sweeps", () => {
    const pSweep = buildParameterScaleSweep(DEFAULT_FORMULA_LAB_INPUT);
    const cSweep = buildContextScaleSweep(DEFAULT_FORMULA_LAB_INPUT);

    expect(pSweep.length).toBe(12);
    expect(pSweep[0].paramsB).toBe(0.5);
    expect(pSweep[pSweep.length - 1].paramsB).toBe(2000);

    expect(cSweep.length).toBe(9);
    expect(cSweep[0].contextLength).toBe(4096);
    expect(cSweep[cSweep.length - 1].contextLength).toBe(2097152);
  });
});
