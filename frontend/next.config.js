/** @type {import('next').NextConfig} */
const nextConfig = {
  // No experimental flags — keep things boring and stable.
  reactStrictMode: true,
  // Backend URL is read from NEXT_PUBLIC_API_URL via lib/api.ts. We
  // don't proxy through Next so the browser hits FastAPI directly,
  // which makes CORS behavior obvious (and required).
};

module.exports = nextConfig;
