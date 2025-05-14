# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json* ./
RUN npm install --legacy-peer-deps

# Copy application code
COPY . .

# Build the application
RUN npm run build --legacy-peer-deps

# Production stage
FROM nginx:stable-alpine AS production

# Copy built assets from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy .env file and env.sh script
COPY .env /usr/share/nginx/html/.env
COPY env.sh /usr/share/nginx/html/env.sh

# Make env.sh executable
RUN chmod +x /usr/share/nginx/html/env.sh

# Expose port
EXPOSE 80

# Start nginx with env.sh
CMD ["/bin/sh", "-c", "cd /usr/share/nginx/html && ./env.sh && nginx -g \"daemon off;\""]
