"use client";

import React from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "@/components/ThemeProvider";

export function ThemeToggleButton() {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === "dark";

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="button button--neutral inline-flex items-center gap-1.5 px-3 py-2 font-mono text-[11px]"
      aria-label={`Switch to ${isDark ? "light" : "dark"} mode`}
      title={`Switch to ${isDark ? "Light Soft UI" : "Dark Soft UI"} theme`}
    >
      {isDark ? (
        <Sun className="h-3.5 w-3.5 text-[#FBBF24]" />
      ) : (
        <Moon className="h-3.5 w-3.5 text-[#2563EB]" />
      )}
      <span>{isDark ? "Light" : "Dark"}</span>
    </button>
  );
}
