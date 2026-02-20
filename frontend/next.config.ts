import type { NextConfig } from "next";

const isGitHubPages = process.env.GITHUB_PAGES === 'true';

const nextConfig: NextConfig = {
  output: 'export',
  basePath: isGitHubPages ? '/img2art-search' : '',
  assetPrefix: isGitHubPages ? '/img2art-search/' : '',
  images: {
    unoptimized: true,
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
      },
      {
        protocol: 'https',
        hostname: 'upload.wikimedia.org',
      },
      {
        protocol: 'https',
        hostname: '*.wikiart.org',
      },
      {
        protocol: 'https',
        hostname: 'chicelli-img2art-search.hf.space',
      },
    ],
  },
  trailingSlash: true,
};

export default nextConfig;
