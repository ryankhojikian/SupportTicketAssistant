# Frontend: build the Vite bundle and serve it via `vite preview`.
# No nginx — `vite preview` also proxies /api/* to the backend container
# (see preview.proxy in vite.config.ts, driven by VITE_DEV_API_PROXY).
FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY . .
RUN npm run build

EXPOSE 4173
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "4173"]
