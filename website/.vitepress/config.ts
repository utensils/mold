import { defineConfig } from 'vitepress'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  title: 'mold',
  description:
    'CLI-native local AI image and video generation for people, scripts, and agents. CUDA, Metal, desktop, web, REST, SSE, and MCP',
  base: '/mold/',

  vite: {
    plugins: [tailwindcss()],
    server: {
      allowedHosts: true,
    },
  },

  head: [
    ['link', { rel: 'icon', href: '/mold/logo-transparent.png' }],
    ['meta', { property: 'og:type', content: 'website' }],
    [
      'meta',
      {
        property: 'og:title',
        content: 'mold: CLI-native local AI image and video generation',
      },
    ],
    [
      'meta',
      {
        property: 'og:description',
        content:
          'CLI-native local AI image and video generation for people, scripts, and agents.',
      },
    ],
    [
      'meta',
      {
        property: 'og:image',
        content: 'https://utensils.io/mold/screenshots/mold-studio-desktop.png',
      },
    ],
    ['meta', { property: 'og:url', content: 'https://utensils.io/mold/' }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    [
      'meta',
      {
        name: 'twitter:title',
        content: 'mold: CLI-native local AI image and video generation',
      },
    ],
    [
      'meta',
      {
        name: 'twitter:description',
        content:
          'CLI-native local AI image and video generation for people, scripts, and agents.',
      },
    ],
    [
      'meta',
      {
        name: 'twitter:image',
        content: 'https://utensils.io/mold/screenshots/mold-studio-desktop.png',
      },
    ],
  ],

  lastUpdated: true,

  markdown: {
    theme: {
      light: 'catppuccin-latte',
      dark: 'catppuccin-mocha',
    },
  },

  sitemap: {
    hostname: 'https://utensils.io/mold/',
  },

  themeConfig: {
    logo: '/logo-transparent.png',

    nav: [
      { text: 'Guide', link: '/guide/' },
      { text: 'Models', link: '/models/' },
      { text: 'API', link: '/api/' },
      { text: 'Deploy', link: '/deployment/' },
      { text: 'Privacy', link: '/privacy' },
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
            { text: 'Video', link: '/guide/video' },
            { text: 'Terminal UI', link: '/guide/tui' },
            { text: 'Desktop App', link: '/guide/desktop' },
            { text: 'iPhone App', link: '/guide/iphone' },
            { text: 'Android App', link: '/guide/android' },
            { text: 'Machines', link: '/guide/machines' },
            { text: 'Image-to-Image', link: '/guide/img2img' },
            { text: 'Identity Photos', link: '/guide/identity' },
            { text: 'Upscaling', link: '/guide/upscaling' },
            { text: 'Prompt Expansion', link: '/guide/expansion' },
            { text: 'Feature Support', link: '/guide/feature-matrix' },
            { text: 'Remote Workflows', link: '/guide/remote-workflows' },
            { text: 'Performance', link: '/guide/performance' },
            { text: 'Custom Models & LoRA', link: '/guide/custom-models' },
            { text: 'Model Discovery Catalog', link: '/docs/catalog' },
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
            { text: 'FLUX.2', link: '/models/flux2' },
            { text: 'FLUX.1', link: '/models/flux' },
            { text: 'SDXL', link: '/models/sdxl' },
            { text: 'SD 1.5', link: '/models/sd15' },
            { text: 'SD 3.5', link: '/models/sd35' },
            { text: 'Z-Image', link: '/models/z-image' },
            { text: 'Wuerstchen', link: '/models/wuerstchen' },
            { text: 'Qwen-Image', link: '/models/qwen-image' },
            { text: 'LTX Video', link: '/models/ltx2' },
            { text: 'LTX Video 0.9.x', link: '/models/ltx-video' },
            { text: 'MiniMax H3', link: '/models/minimax-h3' },
            { text: 'Wan Video', link: '/models/wan' },
            { text: 'Upscalers', link: '/models/upscalers' },
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
            { text: 'mold runpod CLI', link: '/deployment/runpod-cli' },
            { text: 'mold lambda CLI', link: '/deployment/lambda-cli' },
            { text: 'NixOS', link: '/deployment/nixos' },
          ],
        },
      ],
      '/docs/': [
        {
          text: 'Docs',
          items: [{ text: 'Catalog', link: '/docs/catalog' }],
        },
      ],
    },

    socialLinks: [{ icon: 'github', link: 'https://github.com/utensils/mold' }],

    search: {
      provider: 'local',
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright:
        'Copyright <a href="https://jamesbrink.online/">James Brink</a> and <a href="mailto:jeff.dilley@gmail.com">Jeffrey Dilley</a>',
    },

    editLink: {
      pattern: 'https://github.com/utensils/mold/edit/main/website/:path',
      text: 'Edit this page on GitHub',
    },
  },
})
