"use client";

import React, { startTransition, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { BookOpen, Bot, Calculator, Cpu, RefreshCw } from "lucide-react";
import type {
  CatalogSnapshot,
  HFModelDetail,
  HFModelSummary,
  InferenceConfig,
} from "@/types/inference";
import { getDeviceById } from "@/lib/hardware-catalog";
import {
  calculateInference,
  DEFAULT_INFERENCE_CONFIG,
} from "@/lib/inference-math";
import { GeneratedLogo } from "@/components/GeneratedLogo";
import { ThemeToggleButton } from "@/components/ThemeToggleButton";
import { ModelCalculatorTab } from "@/components/ModelCalculatorTab";
import { HardwareCompatibilityTab } from "@/components/HardwareCompatibilityTab";
import { RoamingMascot } from "@/components/RoamingMascot";

const AIHardwareHelpView = dynamic(
  () => import("@/components/AIHardwareHelpView").then((m) => m.AIHardwareHelpView),
  { ssr: false }
);

const ArchitectureModal3D = dynamic(
  () => import("@/components/ArchitectureModal3D").then((m) => m.ArchitectureModal3D),
  { ssr: false }
);

const DeviceDetailModal = dynamic(
  () => import("@/components/DeviceDetailModal").then((m) => m.DeviceDetailModal),
  { ssr: false }
);

interface WorkbenchShellProps {
  initialCatalog: CatalogSnapshot;
}

type ActiveTab = "calculator" | "hardware" | "help";

export function WorkbenchShell({ initialCatalog }: WorkbenchShellProps) {
  const [catalog, setCatalog] = useState<CatalogSnapshot>(initialCatalog);
  const [activeTab, setActiveTab] = useState<ActiveTab>("calculator");
  const [selectedModel, setSelectedModel] = useState<HFModelSummary>(
    () =>
      initialCatalog.models.find((m) => m.id === "Qwen/Qwen2.5-7B-Instruct") ??
      initialCatalog.models[0]
  );
  const [modelDetail, setModelDetail] = useState<HFModelDetail | null>(null);
  const [failedDetailTargetId, setFailedDetailTargetId] = useState<string | null>(null);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("apple-m4-max-64");
  const [config, setConfig] = useState<InferenceConfig>(DEFAULT_INFERENCE_CONFIG);

  const [visualizerModel, setVisualizerModel] = useState<HFModelSummary | null>(null);
  const [inspectedDeviceId, setInspectedDeviceId] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [mascotEnabled, setMascotEnabled] = useState(true);
  const [nowMs, setNowMs] = useState<number | null>(null);

  const selectedDevice = useMemo(
    () => getDeviceById(selectedDeviceId),
    [selectedDeviceId]
  );

  const inspectedDevice = useMemo(
    () => (inspectedDeviceId ? getDeviceById(inspectedDeviceId) : selectedDevice),
    [inspectedDeviceId, selectedDevice]
  );

  const activeDetailTargetId = visualizerModel?.id ?? selectedModel?.id;
  const isLoadingDetail = Boolean(
    activeDetailTargetId &&
      modelDetail?.id !== activeDetailTargetId &&
      failedDetailTargetId !== activeDetailTargetId
  );

  useEffect(() => {
    const initialTimer = window.setTimeout(() => setNowMs(Date.now()), 0);
    const interval = window.setInterval(() => setNowMs(Date.now()), 60_000);
    return () => {
      window.clearTimeout(initialTimer);
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (!activeDetailTargetId) return;
    let cancelled = false;

    const encodedPath = activeDetailTargetId
      .split("/")
      .map((seg) => encodeURIComponent(seg))
      .join("/");

    fetch(`/api/models/${encodedPath}`)
      .then(async (res) => {
        if (!res.ok) return null;
        return (await res.json()) as HFModelDetail;
      })
      .then((detail) => {
        if (cancelled) return;
        if (detail) {
          setModelDetail(detail);
        } else {
          setFailedDetailTargetId(activeDetailTargetId);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setFailedDetailTargetId(activeDetailTargetId);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeDetailTargetId]);

  const activeEstimate = useMemo(() => {
    const modelForCalc =
      modelDetail && modelDetail.id === selectedModel.id ? modelDetail : selectedModel;
    return calculateInference(modelForCalc, selectedDevice, config);
  }, [modelDetail, selectedModel, selectedDevice, config]);

  const handleSelectModel = (model: HFModelSummary) => {
    setSelectedModel(model);
    if (!catalog.models.some((m) => m.id === model.id)) {
      setCatalog((prev) => ({
        ...prev,
        models: [model, ...prev.models],
        count: prev.count + 1,
      }));
    }
  };

  const handleRefreshCatalog = async () => {
    if (isRefreshing) return;
    setIsRefreshing(true);
    try {
      const res = await fetch("/api/models/refresh", {
        method: "POST",
        headers: { "x-monolith-ui": "1" },
      });
      if (res.ok) {
        const nextSnap = (await res.json()) as CatalogSnapshot;
        setCatalog(nextSnap);
      }
    } finally {
      setIsRefreshing(false);
    }
  };

  const nextSyncText = useMemo(() => {
    if (nowMs === null) return "Nightly sync @ 02:00 UTC";
    const diffMs = new Date(catalog.nextScheduledRefreshAt).getTime() - nowMs;
    if (!Number.isFinite(diffMs) || diffMs <= 0) return "Scheduled tonight";
    const hrs = Math.floor(diffMs / 3_600_000);
    const mins = Math.floor((diffMs % 3_600_000) / 60_000);
    return `Nightly sync in ${hrs}h ${mins}m`;
  }, [catalog.nextScheduledRefreshAt, nowMs]);

  const modalTargetModel = useMemo(() => {
    const base = visualizerModel ?? selectedModel;
    if (modelDetail && modelDetail.id === base.id) return modelDetail;
    return base;
  }, [visualizerModel, selectedModel, modelDetail]);

  return (
    <div className="relative z-10 min-h-screen pb-16">
      {/* 95%-WIDTH SOFT UI EVOLUTION HEADER */}
      <header className="sticky top-0 z-30 border-b border-[var(--soft-border)] bg-[var(--soft-header)] backdrop-blur-xl">
        <div className="mx-auto flex w-[95%] max-w-[95vw] flex-wrap items-center justify-between gap-4 py-3.5">
          <GeneratedLogo />

          {/* 3 Primary Tabs: Calculator | Hardware Fit | AI Hardware Help */}
          <nav
            aria-label="Primary workbench views"
            className="well flex flex-wrap items-center gap-1.5 rounded-2xl p-1.5"
          >
            <button
              type="button"
              onClick={() => startTransition(() => setActiveTab("calculator"))}
              className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-semibold transition duration-200 ${
                activeTab === "calculator"
                  ? "button button--primary px-4 py-2"
                  : "muted hover:text-[var(--soft-text)]"
              }`}
            >
              <Calculator className="h-3.5 w-3.5" />
              <span>Model Calculator</span>
            </button>

            <button
              type="button"
              onClick={() => startTransition(() => setActiveTab("hardware"))}
              className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-semibold transition duration-200 ${
                activeTab === "hardware"
                  ? "button button--primary px-4 py-2"
                  : "muted hover:text-[var(--soft-text)]"
              }`}
            >
              <Cpu className="h-3.5 w-3.5" />
              <span>Hardware Fit</span>
            </button>

            <button
              type="button"
              onClick={() => startTransition(() => setActiveTab("help"))}
              className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-semibold transition duration-200 ${
                activeTab === "help"
                  ? "button button--primary px-4 py-2"
                  : "muted hover:text-[var(--soft-text)]"
              }`}
            >
              <BookOpen className="h-3.5 w-3.5" />
              <span>AI Hardware Help</span>
            </button>
          </nav>

          {/* Nightly Sync Pill, Theme Switcher & Mascot Control */}
          <div className="flex flex-wrap items-center gap-2.5">
            <div className="well hidden items-center gap-2 rounded-xl px-3 py-1.5 font-mono text-[11px] text-[var(--soft-muted)] xl:flex">
              <span className="h-2 w-2 rounded-full bg-[var(--soft-green)]" />
              <span className="tabular font-semibold text-[var(--soft-text)]">
                {catalog.count} HF models
              </span>
              <span>·</span>
              <span>{nextSyncText}</span>
            </div>

            <button
              type="button"
              onClick={handleRefreshCatalog}
              disabled={isRefreshing}
              className="button inline-flex items-center gap-1.5 px-3 py-2 font-mono text-[11px]"
              title={`Last synced: ${new Date(catalog.lastRefreshedAt).toLocaleString()}`}
              aria-label="Sync Hugging Face model catalog"
            >
              <RefreshCw
                className={`h-3.5 w-3.5 ${isRefreshing ? "animate-spin" : ""}`}
              />
              <span>{isRefreshing ? "Syncing..." : "Sync HF"}</span>
            </button>

            <ThemeToggleButton />

            <button
              type="button"
              onClick={() => setMascotEnabled((v) => !v)}
              className={`button inline-flex items-center gap-1.5 px-3 py-2 font-mono text-[11px] ${
                mascotEnabled ? "button--primary" : "button--neutral"
              }`}
              title="Toggle roaming M-01 companion mascot"
              aria-label="Toggle M-01 roaming mascot"
            >
              <Bot className="h-3.5 w-3.5" />
              <span>M-01</span>
            </button>
          </div>
        </div>
      </header>

      {/* 95%-WIDTH MAIN WORKSPACE */}
      <main className="mx-auto mt-6 w-[95%] max-w-[95vw]">
        {activeTab === "calculator" && (
          <ModelCalculatorTab
            models={catalog.models}
            selectedModel={selectedModel}
            modelDetail={modelDetail}
            isLoadingDetail={isLoadingDetail}
            selectedDevice={selectedDevice}
            config={config}
            onSelectModel={handleSelectModel}
            onSelectDeviceId={setSelectedDeviceId}
            onOpenDeviceSpecs={() => setInspectedDeviceId(selectedDevice.id)}
            onChangeConfig={setConfig}
            onOpenVisualizer={() => setVisualizerModel(selectedModel)}
          />
        )}

        {activeTab === "hardware" && (
          <HardwareCompatibilityTab
            models={catalog.models}
            selectedDevice={selectedDevice}
            config={config}
            onSelectDeviceId={setSelectedDeviceId}
            onOpenDeviceSpecs={(id) => setInspectedDeviceId(id ?? selectedDevice.id)}
            onChangeConfig={setConfig}
            onInspectModelInCalculator={(model) => {
              handleSelectModel(model);
              setActiveTab("calculator");
            }}
            onVisualizeModel={(model) => {
              handleSelectModel(model);
              setVisualizerModel(model);
            }}
          />
        )}

        {activeTab === "help" && <AIHardwareHelpView />}
      </main>

      {/* 95%-WIDTH SOFT UI EVOLUTION FOOTER */}
      <footer className="mx-auto mt-10 w-[95%] max-w-[95vw] border-t border-[var(--soft-border)] pt-5">
        <div className="muted flex flex-wrap items-center justify-between gap-3 font-mono text-[11px]">
          <span>
            MONOLITH · Next.js + Bun LocalAI Inference Calculator · Live Hugging Face Hub API
          </span>
          <span>
            Last sync: {new Date(catalog.lastRefreshedAt).toUTCString()} ({catalog.source})
          </span>
        </div>
      </footer>

      {/* THREE.JS COLORFUL 3D MODEL ARCHITECTURE MODAL (conditional split chunk) */}
      {visualizerModel && (
        <ArchitectureModal3D
          key={modalTargetModel.id}
          open={Boolean(visualizerModel)}
          model={modalTargetModel}
          device={selectedDevice}
          config={config}
          onChangeConfig={setConfig}
          onOpenDeviceSpecs={() => setInspectedDeviceId(selectedDevice.id)}
          onClose={() => setVisualizerModel(null)}
        />
      )}

      {/* CLICKABLE DEVICE PARAMETERS MODAL (conditional split chunk) */}
      {inspectedDeviceId && (
        <DeviceDetailModal
          open={Boolean(inspectedDeviceId)}
          device={inspectedDevice}
          model={modelDetail && modelDetail.id === selectedModel.id ? modelDetail : selectedModel}
          config={config}
          onChangeConfig={setConfig}
          onOpenHelp={() => startTransition(() => setActiveTab("help"))}
          onClose={() => setInspectedDeviceId(null)}
        />
      )}

      {/* ROAMING MINI MASCOT */}
      <RoamingMascot
        enabled={mascotEnabled}
        modelName={selectedModel?.name}
        deviceName={selectedDevice.name}
        totalRequiredGB={activeEstimate.totalRequiredGB}
        usableGB={activeEstimate.effectiveUsableMemoryGB}
        decodeTps={activeEstimate.estimatedDecodeTps}
        fits={activeEstimate.supported}
      />
    </div>
  );
}
