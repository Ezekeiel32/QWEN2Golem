/** @type {import('next').NextConfig} */
const nextConfig = {
  // Your existing Next.js config
  env: {
    NEXT_PUBLIC_BACKEND_URL: process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5000',
  },
};

export default nextConfig;
