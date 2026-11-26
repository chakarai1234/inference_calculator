import { mkdir, readFile, rename, writeFile } from "fs/promises";
import path from "path";
import type { CatalogSnapshot, HFModelSummary } from "@/types/inference";
import { fetchCatalogFromHF } from "@/lib/hf-client";
import { HF_FALLBACK_SEED } from "@/lib/hf-fallback-seed";

const ONE_DAY_MS = 24 * 60 * 60 * 1000;
const MANUAL_COOLDOWN_MS = 60 * 1000;

interface GlobalSchedulerState {
  memorySnapshot: CatalogSnapshot | null;
  inFlightRefresh: Promise<CatalogSnapshot> | null;
  lastManualRefreshAt: number;
  schedulerStarted: boolean;
  timer?: ReturnType<typeof setTimeout>;
}

const GLOBAL_KEY = "__monolithCatalogState";

function getGlobalState(): GlobalSchedulerState {
  const g = globalThis as unknown as Record<string, GlobalSchedulerState | undefined>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = {
      memorySnapshot: null,
      inFlightRefresh: null,
      lastManualRefreshAt: 0,
      schedulerStarted: false,
    };
  }
  return g[GLOBAL_KEY];
}

function getCacheFilePath(): string {
  return (
    process.env.CATALOG_CACHE_PATH?.trim() ||
    path.join(process.cwd(), ".cache", "hf-catalog.json")
  );
}

function getRefreshHour(): number {
  const parsed = Number(process.env.CATALOG_REFRESH_HOUR ?? 2);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 23) return 2;
  return Math.floor(parsed);
}

export function getNextNightlyRefreshDate(from = new Date()): Date {
  const targetHour = getRefreshHour();
  const next = new Date(from.getTime());
  next.setHours(targetHour, 0, 0, 0);
  if (next.getTime() <= from.getTime() + 5000) {
    next.setDate(next.getDate() + 1);
  }
  return next;
}

function mergeWithSeed(liveModels: HFModelSummary[]): HFModelSummary[] {
  const merged = new Map<string, HFModelSummary>();
  for (const seed of HF_FALLBACK_SEED) {
    merged.set(seed.id, seed);
  }
  for (const live of liveModels) {
    const existing = merged.get(live.id);
    if (!existing) {
      merged.set(live.id, live);
    } else {
      merged.set(live.id, {
        ...live,
        // Preserve rich curated MoE / config hints if HF list response omitted them
        activeParametersB:
          live.isMoE || !existing.isMoE ? live.activeParametersB : existing.activeParametersB,
        isMoE: live.isMoE || existing.isMoE,
        embeddedConfig: live.embeddedConfig ?? existing.embeddedConfig,
      });
    }
  }
  return Array.from(merged.values()).sort((a, b) => b.downloads - a.downloads);
}

async function readDiskSnapshot(): Promise<CatalogSnapshot | null> {
  try {
    const raw = await readFile(/*turbopackIgnore: true*/ getCacheFilePath(), "utf8");
    const parsed = JSON.parse(raw) as CatalogSnapshot;
    if (parsed && Array.isArray(parsed.models) && parsed.models.length > 0) {
      return {
        ...parsed,
        nextScheduledRefreshAt: getNextNightlyRefreshDate().toISOString(),
      };
    }
    return null;
  } catch {
    return null;
  }
}

async function writeDiskSnapshot(snapshot: CatalogSnapshot): Promise<void> {
  const target = getCacheFilePath();
  const dir = path.dirname(target);
  const tempFile = `${target}.${process.pid}.tmp`;
  try {
    await mkdir(dir, { recursive: true });
    await writeFile(tempFile, JSON.stringify(snapshot), "utf8");
    await rename(tempFile, target);
  } catch (err) {
    console.warn("[monolith] Failed to persist HF catalog snapshot:", err);
  }
}

export async function refreshCatalog(
  reason: "scheduled" | "manual" | "stale" = "manual",
  bypassCooldown = false
): Promise<CatalogSnapshot> {
  const state = getGlobalState();

  if (state.inFlightRefresh) {
    return state.inFlightRefresh;
  }

  if (
    reason === "manual" &&
    !bypassCooldown &&
    Date.now() - state.lastManualRefreshAt < MANUAL_COOLDOWN_MS &&
    state.memorySnapshot
  ) {
    return state.memorySnapshot;
  }

  state.inFlightRefresh = (async () => {
    try {
      const liveModels = await fetchCatalogFromHF();
      const now = new Date();
      const models =
        liveModels.length > 0 ? mergeWithSeed(liveModels) : HF_FALLBACK_SEED;

      const snapshot: CatalogSnapshot = {
        models,
        lastRefreshedAt: now.toISOString(),
        nextScheduledRefreshAt: getNextNightlyRefreshDate(now).toISOString(),
        source: liveModels.length > 0 ? "live" : "seed",
        count: models.length,
      };

      state.memorySnapshot = snapshot;
      if (reason === "manual") {
        state.lastManualRefreshAt = Date.now();
      }
      await writeDiskSnapshot(snapshot);
      return snapshot;
    } catch (err) {
      console.warn(`[monolith] Catalog refresh (${reason}) failed:`, err);
      const fallback =
        state.memorySnapshot ??
        (await readDiskSnapshot()) ?? {
          models: HF_FALLBACK_SEED,
          lastRefreshedAt: new Date().toISOString(),
          nextScheduledRefreshAt: getNextNightlyRefreshDate().toISOString(),
          source: "seed" as const,
          count: HF_FALLBACK_SEED.length,
        };
      state.memorySnapshot = fallback;
      return fallback;
    } finally {
      state.inFlightRefresh = null;
    }
  })();

  return state.inFlightRefresh;
}

export async function getCatalog(): Promise<CatalogSnapshot> {
  const state = getGlobalState();

  if (!state.memorySnapshot) {
    const disk = await readDiskSnapshot();
    if (disk) {
      state.memorySnapshot = { ...disk, source: "cache" };
    }
  }

  if (!state.memorySnapshot) {
    // Attempt live fetch on very first load with short grace; fallback safely on error
    return refreshCatalog("stale", true);
  }

  const ageMs = Date.now() - new Date(state.memorySnapshot.lastRefreshedAt).getTime();
  if (ageMs > ONE_DAY_MS || state.memorySnapshot.source === "seed") {
    void refreshCatalog("stale", true);
  }

  return {
    ...state.memorySnapshot,
    nextScheduledRefreshAt: getNextNightlyRefreshDate().toISOString(),
  };
}

export function startNightlyScheduler(): void {
  if (process.env.NEXT_PHASE === "phase-production-build") return;
  const state = getGlobalState();
  if (state.schedulerStarted) return;
  state.schedulerStarted = true;

  const scheduleNext = () => {
    const nextRun = getNextNightlyRefreshDate();
    const delayMs = Math.max(60_000, nextRun.getTime() - Date.now());
    const timer = setTimeout(async () => {
      console.info(
        `[monolith] Running nightly Hugging Face catalog refresh at ${new Date().toISOString()}`
      );
      await refreshCatalog("scheduled", true);
      scheduleNext();
    }, delayMs);

    if (typeof timer.unref === "function") {
      timer.unref();
    }
    state.timer = timer;
  };

  scheduleNext();

  // Warm cache asynchronously on startup if missing or stale
  void (async () => {
    const disk = await readDiskSnapshot();
    if (!disk) {
      await refreshCatalog("stale", true);
      return;
    }
    state.memorySnapshot = { ...disk, source: "cache" };
    const ageMs = Date.now() - new Date(disk.lastRefreshedAt).getTime();
    if (ageMs > ONE_DAY_MS) {
      await refreshCatalog("stale", true);
    }
  })();
}
