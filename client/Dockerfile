FROM node:20 AS builder

WORKDIR /app

COPY . /app

RUN npm install

RUN npx vite build

FROM nginx

COPY --from=builder /app/dist /usr/share/nginx/html

RUN chmod -R 755 /usr/share/nginx/html
