const isVercel = process.env.VERCEL === "1";

/** @type {import('next').NextConfig} */
const nextConfig = {
  // En Vercel: modo server (más seguro para API routes)
  // En desarrollo: modo normal
  // Para exportar estático (e.g. GitHub Pages): output: "export"
  
  images: { unoptimized: true },
  trailingSlash: false,
  allowedDevOrigins: ["http://192.168.100.83:3000", "localhost:3000"],
  outputFileTracingRoot: process.cwd(),
};

export default nextConfig;
