"use client";

import React, { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import Fuse from "fuse.js";
import { Download, Loader2, Search, X } from "lucide-react";
import type { HFModelSummary, ModelFormat } from "@/types/inference";
import { formatCompactNumber, formatParams } from "@/lib/inference-math";

interface ModelFuzzyPickerProps {
  models: HFModelSummary[];
  selectedModel: HFModelSummary;
  onSelectModel: (model: HFModelSummary) => void;
}

type FormatFilter = "all" | ModelFormat;

const FORMAT_FILTERS: { id: FormatFilter; label: string }[] = [
  { id: "all", label: "All Formats" },
  { id: "gguf", label: "GGUF" },
  { id: "safetensors", label: "Safetensors" },
  { id: "mlx", label: "MLX" },
  { id: "diffusers", label: "Diffusers" },
];

const ROW_HEIGHT = 64;
const DEFAULT_VIEWPORT_HEIGHT = 560;
const OVERSCAN_ROWS = 4;

export function ModelFuzzyPicker({
  models,
  selectedModel,
  onSelectModel,
}: ModelFuzzyPickerProps) {
  const [query, setQuery] = useState("");
  const [formatFilter, setFormatFilter] = useState<FormatFilter>("all");
  const [remoteState, setRemoteState] = useState<{
    queryKey: string;
    models: HFModelSummary[];
  }>({ queryKey: "", models: [] });
  const [searchingKey, setSearchingKey] = useState<string | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(DEFAULT_VIEWPORT_HEIGHT);

  const listScrollRef = useRef<HTMLDivElement | null>(null);

  // Measure available container height dynamically so the listbox fills 100% of its card
  useEffect(() => {
    const el = listScrollRef.current;
    if (!el) return;
    const syncHeight = () => {
      const h = el.clientHeight;
      if (h > 0) setViewportHeight(h);
    };
    syncHeight();
    const observer = new ResizeObserver(syncHeight);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const deferredQuery = useDeferredValue(query);

  // Merge catalog models + any live Hub search hits without duplicates
  const combinedModels = useMemo(() => {
    const activeQueryKey = deferredQuery.trim();
    const remoteModels =
      activeQueryKey.length >= 2 && remoteState.queryKey === activeQueryKey
        ? remoteState.models
        : [];
    if (remoteModels.length === 0) return models;
    const seen = new Set(models.map((m) => m.id));
    const extras = remoteModels.filter((m) => !seen.has(m.id));
    return extras.length > 0 ? [...models, ...extras] : models;
  }, [models, remoteState, deferredQuery]);

  // Debounced live Hugging Face Hub API search for arbitrary repo lookups
  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      return;
    }

    const controller = new AbortController();

    const timer = window.setTimeout(async () => {
      setSearchingKey(trimmed);
      try {
        const res = await fetch(
          `/api/models/search?q=${encodeURIComponent(trimmed)}`,
          { signal: controller.signal }
        );
        if (!res.ok) return;
        const payload = (await res.json()) as { models?: HFModelSummary[] };
        if (!controller.signal.aborted && Array.isArray(payload.models)) {
          setRemoteState({ queryKey: trimmed, models: payload.models });
        }
      } catch {
        // Retain local fuzzy catalog gracefully if offline/aborted
      } finally {
        if (!controller.signal.aborted) {
          setSearchingKey(null);
        }
      }
    }, 260);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  const isSearchingHub = Boolean(
    query.trim().length >= 2 && searchingKey === query.trim()
  );

  const fuse = useMemo(
    () =>
      new Fuse(combinedModels, {
        keys: [
          { name: "id", weight: 0.45 },
          { name: "name", weight: 0.3 },
          { name: "author", weight: 0.12 },
          { name: "architectureHint", weight: 0.08 },
          { name: "tags", weight: 0.05 },
        ],
        threshold: 0.36,
        ignoreLocation: true,
        minMatchCharLength: 2,
      }),
    [combinedModels]
  );

  // Return ALL matching models — NO artificial .slice(0, 60) cap
  const filteredModels = useMemo(() => {
    const trimmed = deferredQuery.trim();
    const searched =
      trimmed.length > 0
        ? fuse.search(trimmed).map((hit) => hit.item)
        : combinedModels;

    if (formatFilter === "all") return searched;
    return searched.filter(
      (m) => m.primaryFormat === formatFilter || m.formats.includes(formatFilter)
    );
  }, [deferredQuery, formatFilter, fuse, combinedModels]);

  const resetScrollPosition = () => {
    setScrollTop(0);
    if (listScrollRef.current) {
      listScrollRef.current.scrollTop = 0;
    }
  };

  const handleQueryChange = (nextQuery: string) => {
    setQuery(nextQuery);
    resetScrollPosition();
  };

  const handleFormatChange = (nextFormat: FormatFilter) => {
    setFormatFilter(nextFormat);
    resetScrollPosition();
  };

  const totalCount = filteredModels.length;
  const totalHeight = totalCount * ROW_HEIGHT;
  const visibleCount = Math.ceil(viewportHeight / ROW_HEIGHT);
  const startIndex = Math.max(
    0,
    Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN_ROWS
  );
  const endIndex = Math.min(
    totalCount,
    Math.floor(scrollTop / ROW_HEIGHT) + visibleCount + OVERSCAN_ROWS
  );
  const visibleSlice = filteredModels.slice(startIndex, endIndex);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      {/* Fuzzy Search Input */}
      <div className="well flex shrink-0 items-center gap-2 rounded-xl px-3.5 py-2.5">
        <Search className="h-4 w-4 shrink-0 text-[var(--soft-blue)]" />
        <input
          type="search"
          value={query}
          onChange={(e) => handleQueryChange(e.target.value)}
          placeholder="Fuzzy search ALL Hugging Face models (e.g. Qwen3, Llama 3.3, Flux, GGUF, MLX)..."
          aria-label="Fuzzy search Hugging Face models"
          className="w-full bg-transparent text-xs font-medium text-[var(--soft-text)] placeholder:text-[var(--soft-muted)] focus:outline-none"
        />
        {isSearchingHub && (
          <Loader2
            className="h-3.5 w-3.5 shrink-0 animate-spin text-[var(--soft-blue)]"
            aria-label="Searching Hugging Face Hub"
          />
        )}
        {query && (
          <button
            type="button"
            onClick={() => handleQueryChange("")}
            className="muted rounded-lg p-1 hover:text-[var(--soft-text)]"
            aria-label="Clear model search"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {/* Format Filter Pills + Total Model Counter */}
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap gap-1.5">
          {FORMAT_FILTERS.map((f) => {
            const active = formatFilter === f.id;
            return (
              <button
                key={f.id}
                type="button"
                onClick={() => handleFormatChange(f.id)}
                className={`rounded-lg px-2.5 py-1 text-[11px] font-semibold transition ${
                  active
                    ? "button button--primary px-2.5 py-1"
                    : "button button--neutral px-2.5 py-1"
                }`}
              >
                {f.label}
              </button>
            );
          })}
        </div>

        <span className="muted font-mono text-[11px] tabular">
          Showing {totalCount} of {combinedModels.length} models
        </span>
      </div>

      {/* Virtualized Fixed-Row Listbox filling 100% of card height without inflating grid row */}
      <div className="relative min-h-[460px] flex-1">
        <div
          ref={listScrollRef}
          onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
          className="well soft-scroll absolute inset-0 overflow-y-auto rounded-xl p-1.5"
        >
        {totalCount === 0 ? (
          <div className="flex h-full flex-col items-center justify-center p-6 text-center">
            <p className="text-sm font-semibold">No matching models found</p>
            <p className="muted mt-1 text-xs">
              Try a broader search term or switch format filters back to All Formats.
            </p>
          </div>
        ) : (
          <ul
            role="listbox"
            aria-label="Hugging Face models list"
            className="relative w-full"
            style={{ height: `${totalHeight}px` }}
          >
            {visibleSlice.map((model, offset) => {
              const absoluteIndex = startIndex + offset;
              const isSelected = model.id === selectedModel.id;

              return (
                <li
                  key={model.id}
                  role="option"
                  aria-selected={isSelected}
                  className="absolute left-0 right-0 px-1"
                  style={{
                    height: `${ROW_HEIGHT - 6}px`,
                    transform: `translateY(${absoluteIndex * ROW_HEIGHT}px)`,
                  }}
                >
                  <button
                    type="button"
                    onClick={() => onSelectModel(model)}
                    className={`flex h-full w-full items-center justify-between gap-2.5 rounded-xl px-3 text-left transition ${
                      isSelected
                        ? "button button--primary px-3 py-2"
                        : "surface px-3 py-2 hover:brightness-[1.02]"
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <span className="truncate text-xs font-bold">
                          {model.name}
                        </span>
                        <span
                          className={`shrink-0 rounded px-1.5 py-0.5 font-mono text-[9px] font-bold uppercase ${
                            isSelected
                              ? "bg-white/20 text-white"
                              : "well px-1.5 py-0.5 text-[var(--soft-blue)]"
                          }`}
                        >
                          {model.primaryFormat}
                        </span>
                        {model.isMoE && (
                          <span
                            className={`shrink-0 rounded px-1.5 py-0.5 font-mono text-[9px] font-bold ${
                              isSelected
                                ? "bg-white/20 text-white"
                                : "badge--ok px-1.5 py-0.5"
                            }`}
                          >
                            MoE
                          </span>
                        )}
                      </div>
                      <p
                        className={`truncate font-mono text-[10px] ${
                          isSelected ? "text-white/80" : "muted"
                        }`}
                      >
                        {model.id}
                      </p>
                    </div>

                    <div
                      className={`shrink-0 text-right font-mono text-[11px] tabular ${
                        isSelected ? "text-white" : ""
                      }`}
                    >
                      <p className="font-bold">
                        {formatParams(model.parametersB)}
                      </p>
                      <p
                        className={`inline-flex items-center gap-1 text-[10px] ${
                          isSelected ? "text-white/80" : "muted"
                        }`}
                      >
                        <Download className="h-2.5 w-2.5" />
                        {formatCompactNumber(model.downloads)}
                      </p>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
        </div>
      </div>
    </div>
  );
}
