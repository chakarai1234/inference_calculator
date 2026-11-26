import React from "react";
import { getCatalog } from "@/lib/catalog-store";
import { WorkbenchShell } from "@/components/WorkbenchShell";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const initialCatalog = await getCatalog();
  return <WorkbenchShell initialCatalog={initialCatalog} />;
}
