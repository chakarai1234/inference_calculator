import React from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { GeneratedLogo } from "@/components/GeneratedLogo";
import { ThemeToggleButton } from "@/components/ThemeToggleButton";
import { AIHardwareHelpView } from "@/components/AIHardwareHelpView";

export default function HelpPage() {
  return (
    <div className="relative z-10 min-h-screen pb-16">
      <header className="sticky top-0 z-30 border-b border-[var(--soft-border)] bg-[var(--soft-header)] backdrop-blur-xl">
        <div className="mx-auto flex w-[95%] max-w-[95vw] flex-wrap items-center justify-between gap-4 py-3.5">
          <GeneratedLogo />

          <div className="flex items-center gap-2.5">
            <Link
              href="/"
              className="button button--primary inline-flex items-center gap-1.5 px-4 py-2 text-xs"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              <span>Back to Calculator</span>
            </Link>
            <ThemeToggleButton />
          </div>
        </div>
      </header>

      <main className="mx-auto mt-6 w-[95%] max-w-[95vw]">
        <AIHardwareHelpView />
      </main>
    </div>
  );
}
