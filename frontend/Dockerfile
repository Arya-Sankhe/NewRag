FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY frontend/package*.json ./
RUN npm ci

# Copy source
COPY frontend/. .

# Expose port
EXPOSE 3000

# Run dev server
CMD ["npm", "run", "dev"]

