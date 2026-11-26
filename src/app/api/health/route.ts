import { NextResponse } from "next/server";
import { getCatalog } from "@/lib/catalog-store";

export const dynamic = "force-dynamic";

export async function GET() {
  const snapshot = await getCatalog();
  return NextResponse.json(
    {
      status: "ok",
      uptimeSeconds: Math.round(process.uptime()),
      catalogCount: snapshot.count,
      catalogSource: snapshot.source,
      lastRefreshedAt: snapshot.lastRefreshedAt,
      nextScheduledRefreshAt: snapshot.nextScheduledRefreshAt,
    },
    { headers: { "Cache-Control": "no-store" } }
  );
}
