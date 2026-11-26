import { NextRequest, NextResponse } from "next/server";
import { refreshCatalog } from "@/lib/catalog-store";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  const cronSecret = process.env.CRON_SECRET?.trim();
  const authHeader = request.headers.get("authorization") ?? "";
  const isCronAuthorized =
    Boolean(cronSecret) && authHeader === `Bearer ${cronSecret}`;

  // If request comes from a non-UI external caller when CRON_SECRET is set, enforce secret unless explicit UI flag?
  // Note: If CRON_SECRET is set, require either matching Bearer token OR same-origin UI trigger header;
  // external cron endpoints authenticate cleanly with Authorization: Bearer <CRON_SECRET>.
  if (cronSecret && !isCronAuthorized && request.headers.get("x-monolith-ui") !== "1") {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const snapshot = await refreshCatalog(
    isCronAuthorized ? "scheduled" : "manual",
    isCronAuthorized
  );
  return NextResponse.json(snapshot, {
    headers: { "Cache-Control": "no-store" },
  });
}
