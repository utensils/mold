import { defineConfig } from 'vitepress'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  title: 'mold',
  description:
    'Local AI image generation CLI — FLUX, SD3.5, SD 1.5, SDXL, Z-Image, Flux.2, Qwen-Image & Wuerstchen on your GPU',
  base: '/mold/',

  vite: {
    plugins: [tailwindcss()],
    server: {
      allowedHosts: true,
    },
  },

  head: [['link', { rel: 'icon', href: '/mold/logo-transparent.png' }]],

  lastUpdated: true,

  markdown: {
    theme: {
      light: 'catppuccin-latte',
      dark: 'catppuccin-mocha',
    },
  },

  sitemap: {
    hostname: 'https://utensils.github.io/mold/',
  },

  themeConfig: {
    logo: '/logo-transparent.png',

    nav: [
      { text: 'Guide', link: '/guide/' },
      { text: 'Models', link: '/models/' },
      { text: 'API', link: '/api/' },
      { text: 'Deploy', link: '/deployment/' },
      { text: 'GitHub', link: 'https://github.com/utensils/mold' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Introduction', link: '/guide/' },
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Configuration', link: '/guide/configuration' },
          ],
        },
        {
          text: 'Usage',
          items: [
            { text: 'Generating Images', link: '/guide/generating' },
            { text: 'Image-to-Image', link: '/guide/img2img' },
            { text: 'Prompt Expansion', link: '/guide/expansion' },
            { text: 'Feature Support', link: '/guide/feature-matrix' },
            { text: 'Remote Workflows', link: '/guide/remote-workflows' },
            { text: 'Performance', link: '/guide/performance' },
            { text: 'Custom Models & LoRA', link: '/guide/custom-models' },
            { text: 'Troubleshooting', link: '/guide/troubleshooting' },
            { text: 'OpenClaw', link: '/guide/openclaw' },
            { text: 'CLI Reference', link: '/guide/cli-reference' },
          ],
        },
      ],
      '/models/': [
        {
          text: 'Models',
          items: [
            { text: 'Overview', link: '/models/' },
            { text: 'FLUX.1', link: '/models/flux' },
            { text: 'SDXL', link: '/models/sdxl' },
            { text: 'SD 1.5', link: '/models/sd15' },
            { text: 'SD 3.5', link: '/models/sd35' },
            { text: 'Z-Image', link: '/models/z-image' },
            { text: 'Flux.2 Klein', link: '/models/flux2' },
            { text: 'Wuerstchen', link: '/models/wuerstchen' },
            { text: 'Qwen-Image', link: '/models/qwen-image' },
          ],
        },
      ],
      '/api/': [
        {
          text: 'Server API',
          items: [
            { text: 'REST API', link: '/api/' },
            { text: 'Discord Bot', link: '/api/discord' },
          ],
        },
      ],
      '/deployment/': [
        {
          text: 'Deployment',
          items: [
            { text: 'Overview', link: '/deployment/' },
            { text: 'Docker & RunPod', link: '/deployment/docker' },
            { text: 'NixOS', link: '/deployment/nixos' },
          ],
        },
      ],
    },

    socialLinks: [{ icon: 'github', link: 'https://github.com/utensils/mold' }],

    search: {
      provider: 'local',
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright <a href="https://jamesbrink.online/">James Brink</a>',
    },

    editLink: {
      pattern: 'https://github.com/utensils/mold/edit/main/website/:path',
      text: 'Edit this page on GitHub',
    },
  },
})
