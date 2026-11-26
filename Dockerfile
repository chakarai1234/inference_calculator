# syntax=docker/dockerfile:1.7

FROM oven/bun:1-slim AS base
WORKDIR /app

# Stage 1: Install dependencies cleanly with frozen lockfile
FROM base AS deps
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

# Stage 2: Build Next.js standalone bundle (Node 22 + Bun ensures reliable SWC & worker builds on amd64/arm64)
FROM node:22-bookworm-slim AS builder
WORKDIR /app
COPY --from=base /usr/local/bin/bun /usr/local/bin/bun
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
RUN bun run build

# Stage 3: Minimal production runtime running Next.js 16 standalone under Node 22
FROM node:22-bookworm-slim AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"
ENV CATALOG_CACHE_PATH="/app/data/hf-catalog.json"
ENV CATALOG_REFRESH_HOUR="2"

RUN groupadd --system --gid 1001 nextjs && \
    useradd --system --uid 1001 --gid nextjs --create-home nextjs && \
    mkdir -p /app/data && \
    chown -R nextjs:nextjs /app/data

COPY --from=builder --chown=nextjs:nextjs /app/public ./public
COPY --from=builder --chown=nextjs:nextjs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nextjs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["node", "-e", "fetch('http://127.0.0.1:3000/api/health').then((r)=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"]

CMD ["node", "server.js"]
