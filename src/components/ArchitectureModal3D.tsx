"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import {
  Activity,
  ArrowRight,
  Box,
  ChevronLeft,
  ChevronRight,
  Cpu,
  GitBranch,
  Layers,
  MessageSquare,
  Pause,
  Play,
  RotateCcw,
  Send,
  SkipBack,
  SkipForward,
  SlidersHorizontal,
  Sparkles,
  X,
} from "lucide-react";
import type {
  HardwareDevice,
  HFModelDetail,
  HFModelSummary,
  InferenceConfig,
} from "@/types/inference";
import {
  buildArchitectureBlueprint,
  buildSingleLayerBreakdown,
  type ArchitectureStageNode,
} from "@/lib/architecture-graph";
import {
  calculateInference,
  formatCompactNumber,
  formatParams,
} from "@/lib/inference-math";
import {
  buildPromptFlowSimulation,
  type PromptFlowStep,
} from "@/lib/prompt-flow";
import { useTheme } from "@/components/ThemeProvider";

interface ArchitectureModal3DProps {
  open: boolean;
  model: HFModelSummary | HFModelDetail;
  device: HardwareDevice;
  config: InferenceConfig;
  onChangeConfig: (next: InferenceConfig) => void;
  onOpenDeviceSpecs?: () => void;
  onClose: () => void;
}

type InspectorTab = "flow" | "layer" | "stage" | "anatomy" | "fit" | "config";

interface RendererStats {
  geometries: number;
  textures: number;
  calls: number;
}

const CONTEXT_PRESETS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];
const PROMPT_PRESETS = [
  "Why is the sky blue?",
  "How does AI run on Mac?",
  "Write Python sort",
];
const MAX_FLOATING_WORDS = 5;

const ARCHETYPE_BADGE_LABELS: Record<string, string> = {
  "dense-transformer": "Dense Causal Transformer",
  "moe-transformer": "Sparse Mixture of Experts (MoE)",
  "multimodal-vlm": "Multimodal Vision-Language",
  diffusion: "Latent Diffusion Pipeline",
};

export function ArchitectureModal3D({
  open,
  model,
  device,
  config,
  onChangeConfig,
  onOpenDeviceSpecs,
  onClose,
}: ArchitectureModal3DProps) {
  const { theme } = useTheme();
  const mountRef = useRef<HTMLDivElement | null>(null);
  const wordChipRefs = useRef<(HTMLSpanElement | null)[]>([]);

  const blueprint = useMemo(
    () => buildArchitectureBlueprint(model, config),
    [model, config]
  );
  const estimate = useMemo(
    () => calculateInference(model, device, config),
    [model, device, config]
  );

  const defaultLayerStage = useMemo(
    () =>
      blueprint.stages.find((stage) => stage.layerStart !== undefined) ??
      blueprint.stages[0],
    [blueprint]
  );

  const [promptDraft, setPromptDraft] = useState("Why is the sky blue?");
  const [submittedPrompt, setSubmittedPrompt] = useState("Why is the sky blue?");
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [paused, setPaused] = useState<boolean>(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
  const [stepAutoPlay, setStepAutoPlay] = useState<boolean>(
    () =>
      !(
        typeof window !== "undefined" &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches
      )
  );
  const [stepSpeed, setStepSpeed] = useState<0.75 | 1 | 1.75>(1);

  const promptFlow = useMemo(
    () => buildPromptFlowSimulation(submittedPrompt, model, blueprint, config),
    [submittedPrompt, model, blueprint, config]
  );

  const currentStep: PromptFlowStep =
    promptFlow.steps[
      Math.min(activeStepIndex, Math.max(0, promptFlow.steps.length - 1))
    ];

  const flowStepKey = `${submittedPrompt}:${activeStepIndex}`;
  const [manualStageSelection, setManualStageSelection] = useState<{
    id: string;
    stepKey: string;
  } | null>(null);
  const selectedStageId =
    manualStageSelection?.stepKey === flowStepKey
      ? manualStageSelection.id
      : currentStep?.stageId ?? defaultLayerStage?.id ?? "token-input";

  const [selectedLayerNumber, setSelectedLayerNumber] = useState<number>(
    defaultLayerStage?.layerStart ?? 1
  );
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("flow");
  const [resetTick, setResetTick] = useState<number>(0);
  const activeSceneKey = `${blueprint.modelId}:${theme}:${resetTick}`;
  const [webglErrorState, setWebglErrorState] = useState<{
    sceneKey: string;
    message: string;
  } | null>(null);
  const webglError =
    webglErrorState?.sceneKey === activeSceneKey ? webglErrorState.message : null;
  const [renderStats, setRenderStats] = useState<RendererStats>({
    geometries: 0,
    textures: 0,
    calls: 0,
  });

  const layerBreakdown = useMemo(
    () => buildSingleLayerBreakdown(model, config, selectedLayerNumber),
    [model, config, selectedLayerNumber]
  );

  const selectedStage: ArchitectureStageNode = useMemo(
    () =>
      blueprint.stages.find((s) => s.id === selectedStageId) ??
      blueprint.stages[0],
    [blueprint.stages, selectedStageId]
  );

  // Refs read by Three.js RAF loop (updated in useEffect, never during render)
  const pausedRef = useRef(paused);
  const selectedStageIdRef = useRef(selectedStageId);
  const currentStepRef = useRef(currentStep);
  const stepSpeedRef = useRef(stepSpeed);
  const flowStepKeyRef = useRef(flowStepKey);
  const stepStartedAtRef = useRef<number>(0);
  const requestSingleRenderRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    pausedRef.current = paused;
    selectedStageIdRef.current = selectedStageId;
    currentStepRef.current = currentStep;
    stepSpeedRef.current = stepSpeed;
    flowStepKeyRef.current = flowStepKey;
    requestSingleRenderRef.current?.();
  }, [paused, selectedStageId, currentStep, stepSpeed, flowStepKey, inspectorTab]);

  useEffect(() => {
    if (!open) return;
    stepStartedAtRef.current = performance.now();
  }, [open, activeStepIndex, submittedPrompt]);

  // Timed step progression interval (zero per-frame React state updates)
  useEffect(() => {
    if (!open || !stepAutoPlay || paused || promptFlow.steps.length <= 1) return;
    const intervalMs = Math.round(1800 / stepSpeed);
    const id = window.setInterval(() => {
      if (document.hidden) return;
      setActiveStepIndex((prev) => (prev + 1) % promptFlow.steps.length);
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [open, stepAutoPlay, paused, stepSpeed, promptFlow.steps.length]);

  // Lock body scroll & close on Escape
  useEffect(() => {
    if (!open) return;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => {
      document.body.style.overflow = prevOverflow;
      window.removeEventListener("keydown", handleKey);
    };
  }, [open, onClose]);

  // Core Three.js Soft UI Floating Studio + Word-Projection Loop (Textures: 0)
  useEffect(() => {
    if (!open) return;
    const container = mountRef.current;
    if (!container) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        powerPreference: "low-power",
      });
    } catch {
      queueMicrotask(() =>
        setWebglErrorState({
          sceneKey: activeSceneKey,
          message:
            "WebGL could not initialize on this browser context. Inspect the Flow and Layer panels on the right.",
        })
      );
      return;
    }

    const isDark = theme === "dark";
    const bgHex = isDark ? 0x1a202c : 0xf0f4f8;
    const trackColor = isDark ? 0x334155 : 0xcbd5e1;

    let canvasWidth = container.clientWidth || 800;
    let canvasHeight = container.clientHeight || 500;

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setSize(canvasWidth, canvasHeight);
    renderer.shadowMap.enabled = false;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = isDark ? 1.08 : 1.02;

    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(bgHex);

    const spineStages = blueprint.stages
      .filter((s) => Math.abs(s.position[2]) < 1.2)
      .sort((a, b) => a.position[0] - b.position[0]);
    const branchStages = blueprint.stages.filter((s) => Math.abs(s.position[2]) >= 1.2);

    // Enclose every floating stage (plus slight bottom slack for autoregressive feedback loop)
    const sceneBox = new THREE.Box3();
    blueprint.stages.forEach((s) => {
      const [x, y, z] = s.position;
      const [w, h, d] = s.dimensions;
      sceneBox.expandByPoint(
        new THREE.Vector3(x - w * 0.5, y - h * 0.5, z - d * 0.5)
      );
      sceneBox.expandByPoint(
        new THREE.Vector3(x + w * 0.5, y + h * 0.5, z + d * 0.5)
      );
    });
    if (spineStages.length >= 2) {
      sceneBox.expandByPoint(new THREE.Vector3(0, sceneBox.min.y - 1.3, 0));
    }
    const center = sceneBox.getCenter(new THREE.Vector3());
    const size = sceneBox.getSize(new THREE.Vector3());

    const aspect = canvasWidth / canvasHeight;
    const fovDeg = 42;
    const vFovRad = THREE.MathUtils.degToRad(fovDeg);
    const hFovRad = 2 * Math.atan(Math.tan(vFovRad / 2) * aspect);
    const fitHeightDist = (size.y * 1.32) / (2 * Math.tan(vFovRad / 2));
    const fitWidthDist = (size.x * 1.16) / (2 * Math.tan(hFovRad / 2));
    const fitDist = Math.max(fitHeightDist, fitWidthDist, 14) + size.z * 0.65;

    const camera = new THREE.PerspectiveCamera(
      fovDeg,
      aspect,
      0.1,
      Math.max(180, fitDist * 5)
    );
    camera.position.set(
      center.x,
      center.y + fitDist * 0.24,
      center.z + fitDist * 0.96
    );

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 5;
    controls.maxDistance = Math.max(65, fitDist * 2.5);
    controls.maxPolarAngle = Math.PI * 0.92;
    controls.target.set(center.x, center.y - 0.1, center.z);
    controls.update();

    const hemiLight = new THREE.HemisphereLight(
      isDark ? 0xe0f2fe : 0xffffff,
      isDark ? 0x1e293b : 0xcbd5e1,
      isDark ? 0.95 : 1.1
    );
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, isDark ? 1.35 : 1.25);
    keyLight.position.set(-14, 18, 14);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x93c5fd, isDark ? 0.65 : 0.48);
    fillLight.position.set(14, -8, -12);
    scene.add(fillLight);

    const geometryCache = new Map<string, THREE.BufferGeometry>();
    const materialCache = new Map<string, THREE.Material>();

    const getRoundedBoxGeo = (w: number, h: number, d: number, radius: number) => {
      const rw = Number(w.toFixed(2));
      const rh = Number(h.toFixed(2));
      const rd = Number(d.toFixed(2));
      const rr = Number(
        Math.min(radius, rw * 0.22, rh * 0.22, rd * 0.22).toFixed(2)
      );
      const key = `rbox:${rw}:${rh}:${rd}:${rr}`;
      let geo = geometryCache.get(key);
      if (!geo) {
        geo = new RoundedBoxGeometry(rw, rh, rd, 3, rr);
        geometryCache.set(key, geo);
      }
      return geo;
    };

    const getStdMaterial = (
      hexColor: number | string,
      opts?: { roughness?: number; metalness?: number; emissiveIntensity?: number }
    ) => {
      const r = opts?.roughness ?? 0.42;
      const m = opts?.metalness ?? 0.08;
      const ei = opts?.emissiveIntensity ?? 0.08;
      const colorObj = new THREE.Color(hexColor);
      const key = `std:${colorObj.getHexString()}:${r}:${m}:${ei}`;
      let mat = materialCache.get(key) as THREE.MeshStandardMaterial | undefined;
      if (!mat) {
        mat = new THREE.MeshStandardMaterial({
          color: colorObj,
          roughness: r,
          metalness: m,
          emissive: colorObj,
          emissiveIntensity: ei,
        });
        materialCache.set(key, mat);
      }
      return mat;
    };

    const clickableMeshes: THREE.Mesh[] = [];
    const stageMaterialPairs: {
      stageId: string;
      mesh: THREE.Mesh;
      mat: THREE.MeshStandardMaterial;
      baseY: number;
    }[] = [];

    const bandMaterial = getStdMaterial(isDark ? 0x1e293b : 0xffffff, {
      roughness: 0.3,
      metalness: 0.05,
      emissiveIntensity: 0.04,
    });

    blueprint.stages.forEach((stage) => {
      const [w, h, d] = stage.dimensions;
      const [x, y, z] = stage.position;

      const stageColor = new THREE.Color(stage.colorHex);
      const stageMat = new THREE.MeshStandardMaterial({
        color: stageColor,
        roughness: 0.36,
        metalness: 0.06,
        emissive: stageColor,
        emissiveIntensity: 0.08,
      });
      materialCache.set(`stage:${stage.id}`, stageMat);

      const stageGeo = getRoundedBoxGeo(w, h, d, 0.16);
      const block = new THREE.Mesh(stageGeo, stageMat);
      block.position.set(x, y, z);
      block.userData = {
        stageId: stage.id,
        layerStart: stage.layerStart,
        layerEnd: stage.layerEnd,
      };
      scene.add(block);
      clickableMeshes.push(block);
      stageMaterialPairs.push({
        stageId: stage.id,
        mesh: block,
        mat: stageMat,
        baseY: y,
      });

      if (stage.layerStart !== undefined && h >= 1.6) {
        const ringGeo = getRoundedBoxGeo(w + 0.08, 0.06, d + 0.08, 0.03);
        [-0.3, 0, 0.3].forEach((offsetRatio) => {
          const band = new THREE.Mesh(ringGeo, bandMaterial);
          band.position.set(x, y + h * offsetRatio, z);
          scene.add(band);
        });
      }
    });

    // Selection Halo
    const haloGeo = new THREE.BoxGeometry(1, 1, 1);
    geometryCache.set("halo-unit", haloGeo);
    const haloMat = new THREE.MeshBasicMaterial({
      color: isDark ? 0x60a5fa : 0x2563eb,
      wireframe: true,
      transparent: true,
      opacity: 0.58,
    });
    materialCache.set("halo-mat", haloMat);
    const selectionHalo = new THREE.Mesh(haloGeo, haloMat);
    selectionHalo.visible = false;
    scene.add(selectionHalo);

    // Signal Conduits between Spine + KV/Vision Branches
    const conduitDefs: {
      from: ArchitectureStageNode;
      to: ArchitectureStageNode;
      colorHex: string;
      archHeight: number;
    }[] = [];

    for (let i = 0; i < spineStages.length - 1; i++) {
      conduitDefs.push({
        from: spineStages[i],
        to: spineStages[i + 1],
        colorHex: spineStages[i].accentHex || "#2563eb",
        archHeight: 0.65,
      });
    }

    branchStages.forEach((branch) => {
      if (spineStages.length === 0) return;
      let nearest = spineStages[0];
      let bestDist = Infinity;
      for (const sp of spineStages) {
        const d = Math.abs(sp.position[0] - branch.position[0]);
        if (d < bestDist) {
          bestDist = d;
          nearest = sp;
        }
      }
      conduitDefs.push({
        from: nearest,
        to: branch,
        colorHex:
          branch.category === "kv-cache" ? "#22c55e" : branch.accentHex || "#0891b2",
        archHeight: 1.15,
      });
    });

    const tubeMat = new THREE.MeshStandardMaterial({
      color: trackColor,
      roughness: 0.55,
      metalness: 0.08,
      transparent: true,
      opacity: 0.72,
    });
    materialCache.set("conduit-tube", tubeMat);

    const curves: {
      curve: THREE.CatmullRomCurve3;
      color: THREE.Color;
      speed: number;
      phaseOffset: number;
    }[] = [];

    conduitDefs.forEach((def, idx) => {
      const p0 = new THREE.Vector3(...def.from.position);
      const p3 = new THREE.Vector3(...def.to.position);
      const midX = (p0.x + p3.x) * 0.5;
      const midY = Math.max(p0.y, p3.y) + def.archHeight;
      const midZ = (p0.z + p3.z) * 0.5;
      const curve = new THREE.CatmullRomCurve3([
        p0,
        new THREE.Vector3(midX, midY, midZ),
        p3,
      ]);

      const tubeGeo = new THREE.TubeGeometry(curve, 16, 0.04, 6, false);
      geometryCache.set(`tube:${idx}`, tubeGeo);
      scene.add(new THREE.Mesh(tubeGeo, tubeMat));

      curves.push({
        curve,
        color: new THREE.Color(def.colorHex),
        speed: 0.38,
        phaseOffset: idx * 0.17,
      });
    });

    // Autoregressive Feedback Arch Conduit (loops generated output word back to Attention/Input)
    let feedbackTubeMat: THREE.MeshStandardMaterial | null = null;
    if (spineStages.length >= 2) {
      const pLast = new THREE.Vector3(
        ...spineStages[spineStages.length - 1].position
      );
      const pFirst = new THREE.Vector3(...spineStages[0].position);
      const underY = Math.min(pLast.y, pFirst.y) - 2.4;
      const feedbackCurve = new THREE.CatmullRomCurve3([
        pLast,
        new THREE.Vector3((pLast.x + pFirst.x) * 0.5, underY, 0.45),
        pFirst,
      ]);
      const fbGeo = new THREE.TubeGeometry(feedbackCurve, 24, 0.052, 6, false);
      geometryCache.set("feedback-tube", fbGeo);
      feedbackTubeMat = new THREE.MeshStandardMaterial({
        color: new THREE.Color("#22c55e"),
        emissive: new THREE.Color("#22c55e"),
        emissiveIntensity: 0.15,
        roughness: 0.38,
        transparent: true,
        opacity: 0.25,
      });
      materialCache.set("feedback-tube-mat", feedbackTubeMat);
      scene.add(new THREE.Mesh(fbGeo, feedbackTubeMat));
    }

    const PACKETS_PER_FLOW = 2;
    const packetCount = Math.max(1, curves.length * PACKETS_PER_FLOW);
    const packetGeo = new THREE.IcosahedronGeometry(0.12, 1);
    geometryCache.set("packet-orb", packetGeo);
    const packetMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    materialCache.set("packet-mat", packetMat);

    const instancedPackets = new THREE.InstancedMesh(
      packetGeo,
      packetMat,
      packetCount
    );
    instancedPackets.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    for (let c = 0; c < curves.length; c++) {
      for (let k = 0; k < PACKETS_PER_FLOW; k++) {
        instancedPackets.setColorAt(c * PACKETS_PER_FLOW + k, curves[c].color);
      }
    }
    if (instancedPackets.instanceColor) {
      instancedPackets.instanceColor.needsUpdate = true;
    }
    scene.add(instancedPackets);

    const stageMap = new Map(blueprint.stages.map((s) => [s.id, s]));
    const raycaster = new THREE.Raycaster();
    const pointerNdc = new THREE.Vector2();
    const tmpPos = new THREE.Vector3();
    const tmpScale = new THREE.Vector3(1, 1, 1);
    const tmpQuat = new THREE.Quaternion();
    const tmpMatrix = new THREE.Matrix4();
    const wordStartVec = new THREE.Vector3();
    const wordEndVec = new THREE.Vector3();
    const wordPosVec = new THREE.Vector3();

    let pointerDownPos: { x: number; y: number } | null = null;
    let hoveredStageId: string | null = null;
    let needsRender = true;
    let frameNumber = 0;
    let elapsedAccumulator = 0;

    const projectWordChips = () => {
      const step = currentStepRef.current;
      if (!step) {
        for (let i = 0; i < MAX_FLOATING_WORDS; i++) {
          const el = wordChipRefs.current[i];
          if (el) el.style.opacity = "0";
        }
        return;
      }

      const fromStage = stageMap.get(step.fromStageId) ?? stageMap.get(step.stageId);
      const toStage = stageMap.get(step.stageId) ?? fromStage;
      if (!fromStage || !toStage) return;

      wordStartVec.set(
        fromStage.position[0],
        fromStage.position[1] + fromStage.dimensions[1] * 0.62,
        fromStage.position[2]
      );
      wordEndVec.set(
        toStage.position[0],
        toStage.position[1] + toStage.dimensions[1] * 0.62,
        toStage.position[2]
      );

      const elapsedMs = performance.now() - stepStartedAtRef.current;
      const durationMs = Math.max(450, 1250 / stepSpeedRef.current);
      const linearT = pausedRef.current
        ? 0.9
        : Math.min(1, Math.max(0, elapsedMs / durationMs));
      // Smoothstep easing for glide
      const easedT = linearT * linearT * (3 - 2 * linearT);
      const arcBoost = Math.sin(easedT * Math.PI) * (step.feedbackLoopActive ? -1.8 : 1.05);

      const words = step.activeWords.slice(0, MAX_FLOATING_WORDS);
      const count = words.length;

      for (let i = 0; i < MAX_FLOATING_WORDS; i++) {
        const el = wordChipRefs.current[i];
        if (!el) continue;
        if (i >= count) {
          el.style.opacity = "0";
          continue;
        }

        wordPosVec.lerpVectors(wordStartVec, wordEndVec, easedT);
        const fan = (i - (count - 1) * 0.5) * 1.15;
        wordPosVec.x += fan * 0.45;
        wordPosVec.y += arcBoost + (i % 2 === 0 ? 0.35 : -0.15);

        wordPosVec.project(camera);
        if (wordPosVec.z > 1) {
          el.style.opacity = "0";
          continue;
        }

        const sx = (wordPosVec.x * 0.5 + 0.5) * canvasWidth;
        const sy = (-wordPosVec.y * 0.5 + 0.5) * canvasHeight;
        el.style.transform = `translate3d(${sx.toFixed(1)}px, ${sy.toFixed(
          1
        )}px, 0) translate(-50%, -50%)`;
        el.style.opacity = "1";
      }
    };

    const updateSelectionVisuals = (elapsedSec: number) => {
      const currentSelectedId = selectedStageIdRef.current;
      const activeStep = currentStepRef.current;

      if (feedbackTubeMat) {
        feedbackTubeMat.opacity = activeStep?.feedbackLoopActive ? 0.92 : 0.22;
        feedbackTubeMat.emissiveIntensity = activeStep?.feedbackLoopActive
          ? 0.65
          : 0.1;
      }

      stageMaterialPairs.forEach(({ stageId, mesh, mat, baseY }) => {
        const isSel = stageId === currentSelectedId;
        const isKvHighlight =
          Boolean(activeStep?.kvWriteActive) &&
          blueprint.stages.find((s) => s.id === stageId)?.category === "kv-cache";
        const isHov = stageId === hoveredStageId;
        mat.emissiveIntensity =
          isSel || isKvHighlight ? 0.36 : isHov ? 0.22 : 0.08;
        const targetLift = isSel ? 0.18 : isHov ? 0.08 : 0;
        mesh.position.y = baseY + targetLift;

        if (isSel) {
          const st = stageMap.get(stageId);
          if (st) {
            selectionHalo.visible = true;
            selectionHalo.position.set(
              mesh.position.x,
              mesh.position.y,
              mesh.position.z
            );
            const pulse = 1.06 + Math.sin(elapsedSec * 3.2) * 0.015;
            selectionHalo.scale.set(
              st.dimensions[0] * pulse,
              st.dimensions[1] * pulse,
              st.dimensions[2] * pulse
            );
          }
        }
      });
    };

    const renderFrame = (elapsedSec: number) => {
      updateSelectionVisuals(elapsedSec);
      controls.update();

      let idx = 0;
      for (let c = 0; c < curves.length; c++) {
        const entry = curves[c];
        for (let k = 0; k < PACKETS_PER_FLOW; k++) {
          const raw = elapsedSec * entry.speed + entry.phaseOffset + k * 0.5;
          const t = Math.min(0.9999, Math.max(0, ((raw % 1) + 1) % 1));
          entry.curve.getPointAt(t, tmpPos);
          tmpMatrix.compose(tmpPos, tmpQuat, tmpScale);
          instancedPackets.setMatrixAt(idx, tmpMatrix);
          idx += 1;
        }
      }
      instancedPackets.instanceMatrix.needsUpdate = true;
      renderer.render(scene, camera);
      projectWordChips();

      if (frameNumber % 20 === 0) {
        setRenderStats({
          geometries: renderer.info.memory.geometries,
          textures: renderer.info.memory.textures,
          calls: renderer.info.render.calls,
        });
      }
    };

    requestSingleRenderRef.current = () => {
      needsRender = true;
      renderFrame(elapsedAccumulator);
      setRenderStats({
        geometries: renderer.info.memory.geometries,
        textures: renderer.info.memory.textures,
        calls: renderer.info.render.calls,
      });
    };

    const handleControlsChange = () => {
      needsRender = true;
    };
    controls.addEventListener("change", handleControlsChange);

    const handlePointerDown = (e: PointerEvent) => {
      pointerDownPos = { x: e.clientX, y: e.clientY };
    };

    const handlePointerMove = (e: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      pointerNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointerNdc, camera);
      const hits = raycaster.intersectObjects(clickableMeshes, false);
      const hitId = (hits[0]?.object.userData?.stageId as string | undefined) ?? null;
      if (hitId !== hoveredStageId) {
        hoveredStageId = hitId;
        renderer.domElement.style.cursor = hitId ? "pointer" : "grab";
        needsRender = true;
      }
    };

    const handlePointerUp = (e: PointerEvent) => {
      if (!pointerDownPos) return;
      const dist = Math.hypot(
        e.clientX - pointerDownPos.x,
        e.clientY - pointerDownPos.y
      );
      pointerDownPos = null;
      if (dist > 6) return;

      const rect = renderer.domElement.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      pointerNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointerNdc, camera);
      const hits = raycaster.intersectObjects(clickableMeshes, false);
      const hit = hits[0]?.object;
      if (!hit) return;

      const clickedStageId = hit.userData?.stageId as string | undefined;
      const clickedLayerStart = hit.userData?.layerStart as number | undefined;
      if (clickedStageId) {
        setManualStageSelection({
          id: clickedStageId,
          stepKey: flowStepKeyRef.current,
        });
        if (clickedLayerStart !== undefined) {
          setSelectedLayerNumber(clickedLayerStart);
          setInspectorTab("layer");
        } else {
          setInspectorTab("stage");
        }
        needsRender = true;
      }
    };

    renderer.domElement.addEventListener("pointerdown", handlePointerDown);
    renderer.domElement.addEventListener("pointermove", handlePointerMove);
    renderer.domElement.addEventListener("pointerup", handlePointerUp);

    const resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      if (width <= 0 || height <= 0) return;
      canvasWidth = width;
      canvasHeight = height;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      needsRender = true;
    });
    resizeObserver.observe(container);

    let rafId = 0;
    let lastTs = 0;

    const handleVisibility = () => {
      lastTs = 0;
      if (!document.hidden) needsRender = true;
    };
    document.addEventListener("visibilitychange", handleVisibility);

    const animate = (now: number) => {
      rafId = window.requestAnimationFrame(animate);
      if (document.hidden) return;

      const dt =
        lastTs === 0 ? 0.016 : Math.max(0, Math.min(0.05, (now - lastTs) / 1000));
      lastTs = now;

      if (!pausedRef.current) {
        elapsedAccumulator += dt;
        needsRender = true;
      }

      if (!needsRender) return;
      needsRender = false;
      frameNumber += 1;
      renderFrame(elapsedAccumulator);
    };

    renderFrame(0);
    const statsTimer = window.setTimeout(() => {
      setRenderStats({
        geometries: renderer.info.memory.geometries,
        textures: renderer.info.memory.textures,
        calls: renderer.info.render.calls,
      });
    }, 0);
    rafId = window.requestAnimationFrame(animate);

    return () => {
      window.clearTimeout(statsTimer);
      window.cancelAnimationFrame(rafId);
      requestSingleRenderRef.current = null;
      document.removeEventListener("visibilitychange", handleVisibility);
      resizeObserver.disconnect();
      controls.removeEventListener("change", handleControlsChange);
      renderer.domElement.removeEventListener("pointerdown", handlePointerDown);
      renderer.domElement.removeEventListener("pointermove", handlePointerMove);
      renderer.domElement.removeEventListener("pointerup", handlePointerUp);
      controls.dispose();

      geometryCache.forEach((geo) => geo.dispose());
      geometryCache.clear();
      materialCache.forEach((mat) => mat.dispose());
      materialCache.clear();
      scene.clear();

      renderer.renderLists.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [open, blueprint, theme, resetTick, activeSceneKey]);

  if (!open) return null;

  const handleStageButtonClick = (stage: ArchitectureStageNode) => {
    setManualStageSelection({ id: stage.id, stepKey: flowStepKey });
    if (stage.layerStart !== undefined) {
      setSelectedLayerNumber(stage.layerStart);
      setInspectorTab("layer");
    } else {
      setInspectorTab("stage");
    }
  };

  const handleRunPrompt = (e?: React.FormEvent) => {
    e?.preventDefault();
    const cleaned = promptDraft.trim() || "Why is the sky blue?";
    setPromptDraft(cleaned);
    setSubmittedPrompt(cleaned);
    setActiveStepIndex(0);
    setStepAutoPlay(true);
    setInspectorTab("flow");
    stepStartedAtRef.current = performance.now();
  };

  const cycleSpeed = () => {
    setStepSpeed((s) => (s === 1 ? 1.75 : s === 1.75 ? 0.75 : 1));
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="arch-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/65 p-3 backdrop-blur-md sm:p-5"
    >
      <div className="surface relative flex h-[94vh] w-[96vw] max-w-[1680px] flex-col overflow-hidden p-4 sm:p-5">
        {/* TOP STUDIO HEADER */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--soft-border)] pb-3">
          <div className="flex items-center gap-3">
            <div className="well flex h-10 w-10 items-center justify-center rounded-xl p-0 text-[var(--soft-blue)]">
              <Box className="h-5 w-5" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2
                  id="arch-modal-title"
                  className="text-base font-bold tracking-tight sm:text-lg"
                >
                  {model.name} — 3D Architecture Studio
                </h2>
                <span className="badge--ok rounded-lg px-2.5 py-0.5 font-mono text-[11px] font-semibold">
                  {ARCHETYPE_BADGE_LABELS[blueprint.archetype] ??
                    blueprint.archetype}
                </span>
                <span className="well rounded-lg px-2.5 py-0.5 font-mono text-[11px] font-semibold tabular">
                  {formatParams(model.parametersB)} params ·{" "}
                  {blueprint.architecture.numLayers} layers
                </span>
              </div>
              <p className="muted mt-0.5 text-xs">
                Type any prompt below to watch words tokenize, glide through each 3D block, cache KV states, and generate an answer step-by-step.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setPaused((p) => !p)}
              className="button inline-flex items-center gap-1.5 px-3 py-2 text-xs"
              aria-label={
                paused ? "Resume 3D tensor animation" : "Pause 3D tensor animation"
              }
            >
              {paused ? (
                <Play className="h-3.5 w-3.5" />
              ) : (
                <Pause className="h-3.5 w-3.5" />
              )}
              <span>{paused ? "Play 3D" : "Pause"}</span>
            </button>

            <button
              type="button"
              onClick={() => setResetTick((t) => t + 1)}
              className="button button--neutral inline-flex items-center gap-1.5 px-3 py-2 text-xs"
              aria-label="Reset 3D camera angle"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Reset View</span>
            </button>

            <button
              type="button"
              onClick={onClose}
              className="button button--neutral inline-flex items-center gap-1.5 px-3 py-2 text-xs"
              aria-label="Close 3D Architecture Studio"
            >
              <X className="h-4 w-4" />
              <span>Close</span>
            </button>
          </div>
        </div>

        {/* PROMPT BAR + STEP-BY-STEP WORD MOVEMENT CONTROLS */}
        <form
          onSubmit={handleRunPrompt}
          className="well mt-3 flex flex-wrap items-center justify-between gap-2.5 rounded-2xl px-3.5 py-2.5"
        >
          <div className="flex min-w-[260px] flex-1 items-center gap-2">
            <MessageSquare className="h-4 w-4 shrink-0 text-[var(--soft-blue)]" />
            <input
              id="arch-prompt-input"
              type="text"
              value={promptDraft}
              onChange={(e) => setPromptDraft(e.target.value)}
              placeholder="Type a sentence to trace through the model..."
              aria-label="Prompt sentence to visualize step-by-step word movement"
              className="surface w-full rounded-xl px-3 py-1.5 text-xs font-medium text-[var(--soft-text)] placeholder:text-[var(--soft-muted)] focus:outline-none"
            />
            <button
              type="submit"
              className="button button--primary inline-flex shrink-0 items-center gap-1.5 px-3 py-1.5 text-xs font-bold"
            >
              <Send className="h-3.5 w-3.5" />
              <span>Trace Flow</span>
            </button>
          </div>

          <div className="hidden items-center gap-1 xl:flex">
            {PROMPT_PRESETS.map((sample) => (
              <button
                key={sample}
                type="button"
                onClick={() => {
                  setPromptDraft(sample);
                  setSubmittedPrompt(sample);
                  setActiveStepIndex(0);
                  setStepAutoPlay(true);
                  setInspectorTab("flow");
                }}
                className="button button--neutral rounded-lg px-2 py-1 text-[10px]"
              >
                {sample}
              </button>
            ))}
          </div>

          {/* Step Transport Controls */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span
              className={`rounded-lg px-2 py-1 font-mono text-[10px] font-bold uppercase ${
                currentStep.phase === "prefill" ? "button--primary" : "badge--ok"
              }`}
            >
              {currentStep.phase}
            </span>
            <span className="font-mono text-[11px] font-bold tabular">
              Step {currentStep.stepIndex + 1}/{currentStep.totalSteps}
            </span>

            <button
              type="button"
              onClick={() => {
                setStepAutoPlay(false);
                setActiveStepIndex((i) => Math.max(0, i - 1));
              }}
              disabled={activeStepIndex <= 0}
              className="button button--neutral p-1.5 disabled:opacity-45"
              aria-label="Previous flow step"
            >
              <SkipBack className="h-3.5 w-3.5" />
            </button>

            <button
              type="button"
              onClick={() => setStepAutoPlay((v) => !v)}
              className={`button px-2.5 py-1.5 text-[11px] ${
                stepAutoPlay ? "button--primary px-2.5 py-1.5" : "button--neutral px-2.5 py-1.5"
              }`}
              aria-label={stepAutoPlay ? "Pause step-by-step word trace" : "Play step-by-step word trace"}
            >
              {stepAutoPlay ? "Auto" : "Paused"}
            </button>

            <button
              type="button"
              onClick={() => {
                setStepAutoPlay(false);
                setActiveStepIndex((i) =>
                  (i + 1) % Math.max(1, promptFlow.steps.length)
                );
              }}
              className="button button--neutral p-1.5"
              aria-label="Next flow step"
            >
              <SkipForward className="h-3.5 w-3.5" />
            </button>

            <button
              type="button"
              onClick={cycleSpeed}
              className="button button--neutral px-2 py-1 font-mono text-[11px]"
              aria-label="Cycle word trace speed"
            >
              {stepSpeed}×
            </button>
          </div>
        </form>

        {/* MAIN SPLIT BODY: 3D VIEWPORT + INSPECTOR */}
        <div className="mt-3 grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-12">
          {/* LEFT 7 COLS: FLOATING 3D VIEWPORT + WORD PROJECTION OVERLAY */}
          <div className="well relative flex min-h-[340px] flex-col overflow-hidden p-2 lg:col-span-7">
            <div
              ref={mountRef}
              className="relative h-full min-h-[300px] w-full flex-1 overflow-hidden rounded-xl"
            />

            {/* Floating Projected Word Chips Overlay (0 WebGL textures, direct DOM transform) */}
            <div className="pointer-events-none absolute inset-2 overflow-hidden rounded-xl">
              {Array.from({ length: MAX_FLOATING_WORDS }).map((_, idx) => {
                const word = currentStep.activeWords[idx] ?? "";
                const isDecode = currentStep.phase === "decode";
                return (
                  <span
                    key={idx}
                    ref={(el) => {
                      wordChipRefs.current[idx] = el;
                    }}
                    data-testid={word ? "floating-word-chip" : undefined}
                    className={`absolute top-0 left-0 rounded-xl px-2.5 py-1 font-mono text-[11px] font-bold shadow-md transition-colors ${
                      isDecode
                        ? "bg-[#15803D] text-white"
                        : "bg-[var(--soft-blue-fill)] text-white"
                    }`}
                    style={{
                      opacity: 0,
                      willChange: "transform, opacity",
                    }}
                  >
                    {word}
                  </span>
                );
              })}
            </div>

            {webglError && (
              <div className="absolute inset-4 flex items-center justify-center rounded-xl bg-[var(--soft-bg)]/95 p-6 text-center">
                <p className="muted max-w-md text-sm">{webglError}</p>
              </div>
            )}

            {/* Top clickable stage strip (single clean scrollable pill row) */}
            <div className="soft-scroll pointer-events-auto absolute top-3.5 left-4 right-4 flex flex-nowrap gap-1.5 overflow-x-auto pb-1">
              {blueprint.stages.map((stage) => {
                const active = stage.id === selectedStage.id;
                const hasLayerTag =
                  stage.layerStart !== undefined &&
                  !stage.shortLabel.includes(`L${stage.layerStart}`);
                return (
                  <button
                    key={stage.id}
                    type="button"
                    onClick={() => handleStageButtonClick(stage)}
                    className={`inline-flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1 font-mono text-[11px] font-semibold transition ${
                      active
                        ? "button button--primary px-2.5 py-1"
                        : "button button--neutral px-2.5 py-1"
                    }`}
                  >
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ backgroundColor: stage.colorHex }}
                    />
                    <span>{stage.shortLabel}</span>
                    {hasLayerTag && (
                      <span className="opacity-85">
                        L{stage.layerStart}–{stage.layerEnd}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Bottom Prompt->Answer Live Stream Ribbon + WebGL Telemetry HUD */}
            <div className="pointer-events-none absolute bottom-3 left-4 right-4 space-y-2">
              <div className="surface pointer-events-auto flex flex-wrap items-center gap-2 rounded-xl px-3 py-2 text-xs">
                <span className="muted font-mono text-[10px] font-bold uppercase">
                  Prompt
                </span>
                <div className="flex flex-wrap gap-1">
                  {promptFlow.promptTokens.slice(0, 8).map((tok) => {
                    const isMoving = currentStep.activeWords.includes(tok.text);
                    return (
                      <span
                        key={tok.index}
                        className={`rounded-md px-1.5 py-0.5 font-mono text-[10px] font-semibold ${
                          isMoving
                            ? "bg-[var(--soft-blue-fill)] text-white"
                            : "well px-1.5 py-0.5 text-[var(--soft-text)]"
                        }`}
                      >
                        {tok.text}
                      </span>
                    );
                  })}
                  {promptFlow.promptTokens.length > 8 && (
                    <span className="muted font-mono text-[10px]">
                      +{promptFlow.promptTokens.length - 8}
                    </span>
                  )}
                </div>

                <ArrowRight className="h-3.5 w-3.5 shrink-0 text-[var(--soft-blue)]" />

                <span className="muted font-mono text-[10px] font-bold uppercase">
                  Answer
                </span>
                <div className="flex flex-wrap gap-1">
                  {currentStep.generatedSoFar.length === 0 ? (
                    <span className="muted font-mono text-[10px] italic">
                      prefilling prompt into KV cache...
                    </span>
                  ) : (
                    currentStep.generatedSoFar.map((ans, idx) => (
                      <span
                        key={`${ans}-${idx}`}
                        className="badge--ok rounded-md px-1.5 py-0.5 font-mono text-[10px] font-bold"
                      >
                        {ans}
                      </span>
                    ))
                  )}
                </div>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="surface pointer-events-auto flex items-center gap-2 rounded-xl px-3 py-1.5 font-mono text-[11px] tabular">
                  <Activity className="h-3.5 w-3.5 text-[var(--soft-green)]" />
                  <span>Geoms: {renderStats.geometries}</span>
                  <span className="muted">·</span>
                  <span>Textures: {renderStats.textures}</span>
                  <span className="muted">·</span>
                  <span>Draws: {renderStats.calls}</span>
                </div>

                <div className="surface pointer-events-auto flex items-center gap-2 rounded-xl px-3 py-1.5 font-mono text-[11px] tabular">
                  <span
                    className={
                      estimate.fits ? "stat--up font-bold" : "stat--warn font-bold"
                    }
                  >
                    {estimate.fits ? "FITS ON DEVICE" : "EXCEEDS VRAM"}
                  </span>
                  <span className="muted">·</span>
                  <span>
                    {estimate.totalRequiredGB.toFixed(1)} /{" "}
                    {estimate.effectiveUsableMemoryGB.toFixed(1)} GB
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT 5 COLS: INTERACTIVE BREAKDOWN INSPECTOR */}
          <div className="surface flex min-h-0 flex-col overflow-hidden p-4 lg:col-span-5">
            {/* Inspector Tabs */}
            <div
              role="tablist"
              aria-label="Architecture breakdown inspector tabs"
              className="well grid grid-cols-6 gap-1 rounded-xl p-1"
            >
              {(
                [
                  { id: "flow", label: "Flow", icon: GitBranch },
                  { id: "layer", label: "Layer", icon: Layers },
                  { id: "stage", label: "Stage", icon: Box },
                  { id: "anatomy", label: "Anatomy", icon: Sparkles },
                  { id: "fit", label: "Ctx Fit", icon: Cpu },
                  { id: "config", label: "Config", icon: SlidersHorizontal },
                ] as const
              ).map(({ id, label, icon: Icon }) => {
                const active = inspectorTab === id;
                return (
                  <button
                    key={id}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    onClick={() => setInspectorTab(id)}
                    className={`inline-flex items-center justify-center gap-1 rounded-lg px-1.5 py-2 text-[11px] font-semibold transition ${
                      active
                        ? "button button--primary px-1.5 py-2"
                        : "muted hover:text-[var(--soft-text)]"
                    }`}
                  >
                    <Icon className="h-3.5 w-3.5 shrink-0" />
                    <span>{label}</span>
                  </button>
                );
              })}
            </div>

            {/* Scrollable Inspector Content */}
            <div className="soft-scroll mt-3 min-h-0 flex-1 space-y-3.5 overflow-y-auto pr-1">
              {inspectorTab === "flow" && (
                <>
                  {/* Current Step Card */}
                  <div className="well space-y-2.5 rounded-xl p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span
                        className={`rounded-lg px-2.5 py-0.5 font-mono text-[10px] font-bold uppercase ${
                          currentStep.phase === "prefill"
                            ? "button--primary"
                            : "badge--ok"
                        }`}
                      >
                        {currentStep.phase === "prefill"
                          ? "Phase 1 · Parallel Prefill"
                          : "Phase 2 · Autoregressive Decode"}
                      </span>
                      <span className="muted font-mono text-[11px] tabular">
                        {currentStep.tensorBadge}
                      </span>
                    </div>

                    <h3 className="text-sm font-bold">{currentStep.title}</h3>
                    <p className="muted text-xs leading-relaxed">
                      {currentStep.explanation}
                    </p>
                  </div>

                  {/* Top-4 Next-Word Softmax Probabilities */}
                  {currentStep.candidates && (
                    <div className="well space-y-2 rounded-xl p-3.5">
                      <p className="muted font-mono text-[10px] font-bold uppercase">
                        Next-Word Softmax Probabilities (Sampled Output)
                      </p>
                      {currentStep.candidates.map((cand) => (
                        <div key={cand.token} className="space-y-1">
                          <div className="flex items-center justify-between font-mono text-xs">
                            <span
                              className={
                                cand.chosen
                                  ? "stat--up font-bold"
                                  : "text-[var(--soft-text)]"
                              }
                            >
                              {cand.chosen ? `★ "${cand.token}"` : `"${cand.token}"`}
                            </span>
                            <span className="tabular font-semibold">
                              {cand.probPct}%
                            </span>
                          </div>
                          <div
                            className="meter w-full"
                            style={
                              {
                                "--value": `${cand.probPct}%`,
                                "--meter-color": cand.chosen
                                  ? "var(--soft-green)"
                                  : "var(--soft-blue-fill)",
                              } as React.CSSProperties
                            }
                          />
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Attention Links (Token-to-Token) */}
                  {currentStep.attentionLinks &&
                    currentStep.attentionLinks.length > 0 && (
                      <div className="well space-y-2 rounded-xl p-3.5">
                        <p className="muted font-mono text-[10px] font-bold uppercase">
                          Attention Focus Weights (Query → Cached Keys)
                        </p>
                        {currentStep.attentionLinks.map((lnk, idx) => (
                          <div key={`${lnk.toWord}-${idx}`} className="space-y-1">
                            <div className="flex items-center justify-between font-mono text-xs">
                              <span>
                                &ldquo;{lnk.fromWord}&rdquo; → &ldquo;{lnk.toWord}
                                &rdquo;
                              </span>
                              <span className="tabular font-semibold">
                                {lnk.weightPct}%
                              </span>
                            </div>
                            <div
                              className="meter w-full"
                              style={
                                {
                                  "--value": `${lnk.weightPct}%`,
                                  "--meter-color": "var(--soft-blue-fill)",
                                } as React.CSSProperties
                              }
                            />
                          </div>
                        ))}
                      </div>
                    )}

                  {/* Live KV Cache Memory Contents */}
                  <div className="well space-y-2 rounded-xl p-3.5">
                    <div className="flex items-center justify-between font-mono text-[10px] uppercase">
                      <span className="muted font-bold">
                        Live KV Cache Bank ({currentStep.kvCachedWords.length}{" "}
                        tokens stored)
                      </span>
                      {currentStep.kvWriteActive && (
                        <span className="stat--up font-bold">Writing K/V</span>
                      )}
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {currentStep.kvCachedWords.length === 0 ? (
                        <span className="muted text-xs">
                          Empty until first Attention layer processes prompt.
                        </span>
                      ) : (
                        currentStep.kvCachedWords.map((w, idx) => (
                          <span
                            key={`${w}-${idx}`}
                            className="surface rounded-lg px-2 py-0.5 font-mono text-[11px] font-semibold"
                          >
                            {w}
                          </span>
                        ))
                      )}
                    </div>
                  </div>

                  {/* Clickable Step-by-Step Timeline */}
                  <div className="space-y-1.5">
                    <p className="muted font-mono text-[10px] font-bold uppercase tracking-wider">
                      Word-by-Word Execution Timeline
                    </p>
                    {promptFlow.steps.map((step) => {
                      const isCur = step.stepIndex === activeStepIndex;
                      return (
                        <button
                          key={step.stepIndex}
                          type="button"
                          onClick={() => {
                            setStepAutoPlay(false);
                            setActiveStepIndex(step.stepIndex);
                          }}
                          className={`flex w-full items-center justify-between gap-2 rounded-xl px-3 py-2 text-left text-xs transition ${
                            isCur
                              ? "button button--primary px-3 py-2"
                              : "well px-3 py-2 hover:brightness-[1.02]"
                          }`}
                        >
                          <div className="min-w-0 flex-1">
                            <span className="block truncate font-semibold">
                              {step.stepIndex + 1}. {step.title}
                            </span>
                          </div>
                          <span
                            className={`shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase ${
                              isCur ? "bg-white/20 text-white" : "muted"
                            }`}
                          >
                            {step.phase}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </>
              )}

              {inspectorTab === "layer" && (
                <>
                  {/* Layer Scrubber */}
                  <div className="well space-y-3 rounded-xl p-4">
                    <div className="flex items-center justify-between gap-2">
                      <div>
                        <p className="muted font-mono text-[10px] uppercase tracking-wider">
                          Interactive Per-Layer Breakdown
                        </p>
                        <h3 className="text-base font-bold tabular">
                          Layer {layerBreakdown.layerNumber} of{" "}
                          {layerBreakdown.totalLayers}
                        </h3>
                      </div>

                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() =>
                            setSelectedLayerNumber((n) => Math.max(1, n - 1))
                          }
                          disabled={layerBreakdown.layerNumber <= 1}
                          className="button button--neutral p-2 disabled:opacity-45"
                          aria-label="Previous transformer layer"
                        >
                          <ChevronLeft className="h-4 w-4" />
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            setSelectedLayerNumber((n) =>
                              Math.min(layerBreakdown.totalLayers, n + 1)
                            )
                          }
                          disabled={
                            layerBreakdown.layerNumber >= layerBreakdown.totalLayers
                          }
                          className="button button--neutral p-2 disabled:opacity-45"
                          aria-label="Next transformer layer"
                        >
                          <ChevronRight className="h-4 w-4" />
                        </button>
                      </div>
                    </div>

                    <input
                      type="range"
                      min={1}
                      max={layerBreakdown.totalLayers}
                      step={1}
                      value={layerBreakdown.layerNumber}
                      onChange={(e) =>
                        setSelectedLayerNumber(Number(e.target.value))
                      }
                      className="soft-range w-full"
                      aria-label="Select layer number"
                    />

                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="badge--ok rounded-lg px-2 py-0.5 text-[11px] font-semibold">
                        {layerBreakdown.depthRole}
                      </span>
                      <div className="flex gap-1.5">
                        {[
                          { label: "L1", val: 1 },
                          {
                            label: "Mid",
                            val: Math.max(
                              1,
                              Math.round(layerBreakdown.totalLayers / 2)
                            ),
                          },
                          {
                            label: `L${layerBreakdown.totalLayers}`,
                            val: layerBreakdown.totalLayers,
                          },
                        ].map((preset) => (
                          <button
                            key={preset.label}
                            type="button"
                            onClick={() => setSelectedLayerNumber(preset.val)}
                            className={`rounded-lg px-2.5 py-1 font-mono text-[11px] font-semibold ${
                              layerBreakdown.layerNumber === preset.val
                                ? "button button--primary px-2.5 py-1"
                                : "button button--neutral px-2.5 py-1"
                            }`}
                          >
                            {preset.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Layer Totals Grid */}
                  <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
                    <div className="well rounded-xl p-3">
                      <p className="muted font-mono text-[10px]">Total Params</p>
                      <p className="mt-0.5 font-mono text-sm font-bold tabular">
                        {layerBreakdown.totalLayerParamsLabel}
                      </p>
                    </div>
                    <div className="well rounded-xl p-3">
                      <p className="muted font-mono text-[10px]">Active Params</p>
                      <p className="mt-0.5 font-mono text-sm font-bold tabular text-[var(--soft-blue)]">
                        {layerBreakdown.activeLayerParamsLabel}
                      </p>
                    </div>
                    <div className="well rounded-xl p-3">
                      <p className="muted font-mono text-[10px]">
                        Weights @ {config.quantization}
                      </p>
                      <p className="mt-0.5 font-mono text-sm font-bold tabular">
                        {layerBreakdown.totalLayerMemoryMB.toFixed(1)} MB
                      </p>
                    </div>
                    <div className="well rounded-xl p-3">
                      <p className="muted font-mono text-[10px]">
                        KV (This Layer)
                      </p>
                      <p className="mt-0.5 font-mono text-sm font-bold tabular text-[var(--soft-green-text)]">
                        {layerBreakdown.kvCacheThisLayerMB.toFixed(1)} MB
                      </p>
                    </div>
                  </div>

                  {/* Attn vs FFN Parameter Split Meter */}
                  <div className="well space-y-2 rounded-xl p-3.5">
                    <div className="flex items-center justify-between text-xs font-semibold">
                      <span>Attention vs MLP / MoE Split</span>
                      <span className="muted font-mono tabular">
                        {layerBreakdown.attnSharePct}% Attn ·{" "}
                        {layerBreakdown.ffnSharePct}% FFN ·{" "}
                        {layerBreakdown.layerFlopsPerTokenG.toFixed(2)} GFLOPs/tok
                      </span>
                    </div>
                    <div
                      className="meter w-full"
                      style={
                        {
                          "--value": `${layerBreakdown.attnSharePct}%`,
                          "--meter-color": "var(--soft-blue-fill)",
                        } as React.CSSProperties
                      }
                    />
                  </div>

                  {/* Intra-layer Forward Pass Sub-Steps */}
                  <div className="space-y-2">
                    <p className="muted font-mono text-[11px] uppercase tracking-wider">
                      Single-Layer Forward Pass Sequence (
                      {layerBreakdown.subSteps.length} ops)
                    </p>
                    {layerBreakdown.subSteps.map((step) => (
                      <div key={step.id} className="well rounded-xl p-3">
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span
                              className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg font-mono text-xs font-bold text-white"
                              style={{ backgroundColor: step.colorHex }}
                            >
                              {step.stepNumber}
                            </span>
                            <div>
                              <h4 className="text-xs font-bold">{step.name}</h4>
                              <p className="muted font-mono text-[11px] tabular">
                                {step.tensorShape}
                              </p>
                            </div>
                          </div>
                          <div className="text-right font-mono text-[11px] tabular">
                            <p className="font-semibold">{step.paramsLabel}</p>
                            <p className="muted">{step.memoryMBLabel}</p>
                          </div>
                        </div>
                        <p className="muted mt-1.5 text-xs leading-relaxed">
                          {step.role}
                        </p>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {inspectorTab === "stage" && (
                <div className="space-y-3">
                  <div className="well rounded-xl p-4">
                    <div className="flex items-center gap-2">
                      <span
                        className="h-3.5 w-3.5 rounded-full"
                        style={{ backgroundColor: selectedStage.colorHex }}
                      />
                      <span className="muted font-mono text-[11px] uppercase">
                        {selectedStage.category} Stage
                      </span>
                    </div>
                    <h3 className="mt-1 text-base font-bold">
                      {selectedStage.label}
                    </h3>
                    <p className="muted mt-1.5 text-xs leading-relaxed">
                      {selectedStage.description}
                    </p>

                    <div className="mt-3 grid grid-cols-2 gap-2.5 font-mono text-xs">
                      <div className="surface rounded-xl p-2.5">
                        <span className="muted block text-[10px]">
                          Tensor Shape
                        </span>
                        <span className="mt-0.5 block font-semibold tabular">
                          {selectedStage.tensorShape}
                        </span>
                      </div>
                      <div className="surface rounded-xl p-2.5">
                        <span className="muted block text-[10px]">
                          Stage Params / Memory
                        </span>
                        <span className="mt-0.5 block font-semibold tabular">
                          {selectedStage.parameterCountLabel} ·{" "}
                          {selectedStage.memoryLabel}
                        </span>
                      </div>
                      <div className="surface col-span-2 rounded-xl p-2.5">
                        <span className="muted block text-[10px]">
                          Compute Equation
                        </span>
                        <span className="mt-0.5 block font-semibold">
                          {selectedStage.formula}
                        </span>
                      </div>
                    </div>

                    {selectedStage.layerStart !== undefined && (
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedLayerNumber(selectedStage.layerStart ?? 1);
                          setInspectorTab("layer");
                        }}
                        className="button button--primary mt-3 w-full py-2.5 text-xs"
                      >
                        Inspect Layers {selectedStage.layerStart}–
                        {selectedStage.layerEnd} Sub-Steps
                      </button>
                    )}
                  </div>
                </div>
              )}

              {inspectorTab === "anatomy" && (
                <div className="space-y-2.5">
                  <div className="well rounded-xl p-4">
                    <h3 className="text-sm font-bold">
                      Inference Pipeline & Efficiency Telemetry
                    </h3>
                    <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-xs tabular">
                      <div>
                        <span className="muted block text-[10px]">
                          Compute / Token
                        </span>
                        <span className="font-bold">
                          {blueprint.deepMetrics.totalFlopsPerTokenG.toFixed(1)}{" "}
                          GFLOPs
                        </span>
                      </div>
                      <div>
                        <span className="muted block text-[10px]">
                          Attention Ratio
                        </span>
                        <span className="font-bold">
                          {blueprint.metrics.gqaRatioLabel}
                        </span>
                      </div>
                      <div>
                        <span className="muted block text-[10px]">
                          KV / Token
                        </span>
                        <span className="font-bold">
                          {blueprint.metrics.kvPerTokenKB} KB/tok
                        </span>
                      </div>
                      <div>
                        <span className="muted block text-[10px]">
                          GQA KV Saved
                        </span>
                        <span className="stat--up font-bold">
                          {blueprint.deepMetrics.kvSavedByGqaPct}%
                        </span>
                      </div>
                    </div>
                  </div>

                  {blueprint.componentBreakdown.map((item) => (
                    <div
                      key={item.id}
                      className="well space-y-1.5 rounded-xl p-3.5"
                    >
                      <div className="flex items-center justify-between text-xs font-semibold">
                        <span>{item.name}</span>
                        <span className="font-mono tabular">
                          {item.pctOfTotal}% · {item.paramsLabel} (
                          {item.memoryGB.toFixed(2)} GB)
                        </span>
                      </div>
                      <div
                        className="meter w-full"
                        style={
                          {
                            "--value": `${Math.max(4, item.pctOfTotal)}%`,
                            "--meter-color": item.colorHex,
                          } as React.CSSProperties
                        }
                      />
                      <p className="muted text-[11px]">{item.detail}</p>
                    </div>
                  ))}
                </div>
              )}

              {inspectorTab === "fit" && (
                <div className="space-y-3">
                  <div className="well space-y-3 rounded-xl p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="muted font-mono text-[10px] uppercase">
                          Active Accelerator
                        </p>
                        <h3 className="text-sm font-bold">{device.name}</h3>
                      </div>
                      {onOpenDeviceSpecs && (
                        <button
                          type="button"
                          onClick={onOpenDeviceSpecs}
                          className="button px-3 py-1.5 text-xs"
                        >
                          Device Specs
                        </button>
                      )}
                    </div>

                    <div>
                      <div className="mb-1 flex items-center justify-between text-xs font-semibold">
                        <span>Context Window</span>
                        <span className="font-mono tabular text-[var(--soft-blue)]">
                          {config.contextLength.toLocaleString()} tokens
                        </span>
                      </div>
                      <input
                        type="range"
                        min={512}
                        max={Math.max(
                          131072,
                          blueprint.architecture.maxContextLength
                        )}
                        step={512}
                        value={config.contextLength}
                        onChange={(e) =>
                          onChangeConfig({
                            ...config,
                            contextLength: Number(e.target.value),
                          })
                        }
                        className="soft-range w-full"
                        aria-label="Adjust context window tokens inside 3D studio"
                      />
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {CONTEXT_PRESETS.map((ctx) => (
                          <button
                            key={ctx}
                            type="button"
                            onClick={() =>
                              onChangeConfig({ ...config, contextLength: ctx })
                            }
                            className={`rounded-lg px-2 py-1 font-mono text-[11px] ${
                              config.contextLength === ctx
                                ? "button button--primary px-2 py-1"
                                : "button button--neutral px-2 py-1"
                            }`}
                          >
                            {formatCompactNumber(ctx)}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="well space-y-2 rounded-xl p-4 font-mono text-xs tabular">
                    <div className="flex justify-between">
                      <span className="muted">
                        Model Weights ({config.quantization})
                      </span>
                      <span className="font-semibold">
                        {estimate.weightsGB.toFixed(2)} GB
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="muted">
                        KV Cache ({config.kvCachePrecision})
                      </span>
                      <span className="font-semibold">
                        {estimate.kvCacheGB.toFixed(2)} GB
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="muted">Activations & Runtime</span>
                      <span className="font-semibold">
                        {estimate.overheadGB.toFixed(2)} GB
                      </span>
                    </div>
                    <div className="flex justify-between border-t border-[var(--soft-border)] pt-2 font-bold">
                      <span>Total Footprint vs Usable</span>
                      <span
                        className={estimate.fits ? "stat--up" : "stat--warn"}
                      >
                        {estimate.totalRequiredGB.toFixed(2)} /{" "}
                        {estimate.effectiveUsableMemoryGB.toFixed(2)} GB
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {inspectorTab === "config" && (
                <div className="well grid grid-cols-2 gap-2.5 rounded-xl p-4 font-mono text-xs tabular">
                  <div>
                    <span className="muted block text-[10px]">Architecture</span>
                    <span className="font-bold">
                      {blueprint.architecture.architectureName}
                    </span>
                  </div>
                  <div>
                    <span className="muted block text-[10px]">
                      Config Source
                    </span>
                    <span className="font-bold">
                      {blueprint.architecture.source}
                    </span>
                  </div>
                  <div>
                    <span className="muted block text-[10px]">
                      Hidden Dim (d_model)
                    </span>
                    <span className="font-bold">
                      {blueprint.architecture.hiddenSize}
                    </span>
                  </div>
                  <div>
                    <span className="muted block text-[10px]">
                      FFN Intermediate
                    </span>
                    <span className="font-bold">
                      {blueprint.architecture.intermediateSize}
                    </span>
                  </div>
                  <div>
                    <span className="muted block text-[10px]">
                      Attention Heads (Q / KV)
                    </span>
                    <span className="font-bold">
                      {blueprint.architecture.numAttentionHeads} /{" "}
                      {blueprint.architecture.numKeyValueHeads}
                    </span>
                  </div>
                  <div>
                    <span className="muted block text-[10px]">Vocab Size</span>
                    <span className="font-bold">
                      {blueprint.architecture.vocabSize.toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="muted block text-[10px]">
                      Native Max Context
                    </span>
                    <span className="font-bold">
                      {blueprint.architecture.maxContextLength.toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="muted block text-[10px]">RoPE Theta</span>
                    <span className="font-bold">
                      {blueprint.architecture.ropeTheta.toLocaleString()}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
