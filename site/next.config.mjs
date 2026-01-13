const isProd = process.env.NODE_ENV === "production";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
  ...(isProd ? { basePath: "/Code2_AI", assetPrefix: "/Code2_AI/" } : {}),
  allowedDevOrigins: ["http://192.168.100.83:3000"],
};

export default nextConfig;
