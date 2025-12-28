# Frontend Environment Variables

Create a `.env.local` file with the following variables:

```env
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# Backend WebSocket URL
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

For Docker Compose, these are automatically set via the docker-compose.yml.
