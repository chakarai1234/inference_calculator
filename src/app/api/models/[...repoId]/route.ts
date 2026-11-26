import { NextRequest, NextResponse } from "next/server";
import { getCatalog } from "@/lib/catalog-store";
import { fetchModelDetail, isValidRepoId } from "@/lib/hf-client";

export const dynamic = "force-dynamic";

export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ repoId: string[] }> }
) {
  const { repoId: segments } = await context.params;
  const repoId = (segments ?? []).map((s) => decodeURIComponent(s)).join("/");

  if (!repoId || !isValidRepoId(repoId)) {
    return NextResponse.json(
      { error: "Invalid Hugging Face repository identifier" },
      { status: 400 }
    );
  }

  try {
    const catalog = await getCatalog();
    const summaryFallback = catalog.models.find((m) => m.id === repoId);
    const detail = await fetchModelDetail(repoId, summaryFallback);
    return NextResponse.json(detail, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (err) {
    console.warn(`[monolith] Model detail fetch failed for ${repoId}:`, err);
    return NextResponse.json(
      { error: `Failed to fetch model metadata for ${repoId}` },
      { status: 502 }
    );
  }
}
