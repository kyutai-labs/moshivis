FROM node:20 AS builder

WORKDIR /app

COPY . /app

RUN npm install

RUN npx vite build

FROM scratch AS build_result

COPY --from=builder /app/dist /
