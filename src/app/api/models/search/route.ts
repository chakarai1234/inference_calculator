import { NextRequest, NextResponse } from "next/server";
import { searchHFModels } from "@/lib/hf-client";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const q = request.nextUrl.searchParams.get("q")?.trim() ?? "";
  if (!q || q.length < 2) {
    return NextResponse.json({ models: [] });
  }

  try {
    const models = await searchHFModels(q);
    return NextResponse.json(
      { models },
      { headers: { "Cache-Control": "no-store" } }
    );
  } catch (err) {
    console.warn("[monolith] Live HF search failed:", err);
    return NextResponse.json(
      { models: [], error: "Hugging Face Hub search temporarily unavailable" },
      { status: 502 }
    );
  }
}
