import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */

  // Proxy image requests to the backend
  async rewrites() {
    // Get backend URL from environment or use default
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://backend:8000';

    return [
      {
        source: '/api/v1/images/:path*',
        destination: `${backendUrl}/api/v1/images/:path*`,
      },
    ];
  },
};

export default nextConfig;
