import { NextResponse } from "next/server";
import { getCatalog } from "@/lib/catalog-store";

export const dynamic = "force-dynamic";

export async function GET() {
  const snapshot = await getCatalog();
  return NextResponse.json(snapshot, {
    headers: {
      "Cache-Control": "no-store",
    },
  });
}
