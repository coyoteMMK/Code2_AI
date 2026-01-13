/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },

  // Para GitHub Pages: tu web vive en /NOMBRE_REPO
  // Lo ponemos por variable de entorno para no romper en local
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || "/Code2_AI",
};

module.exports = nextConfig;
