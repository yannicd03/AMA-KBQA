# React demo UI for AMA-KBQA: build with Node, serve the static bundle with
# nginx, which also proxies /api/ to the FastAPI service (`api`, port 8506).
#
#   docker build -t ama-kbqa-web web/

# ── Build ────────────────────────────────────────────────────────────────────
FROM node:24.21.0-alpine3.24 AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY . .
RUN npm run build

# ── Serve ────────────────────────────────────────────────────────────────────
FROM nginx:1.30.4-alpine3.24

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
