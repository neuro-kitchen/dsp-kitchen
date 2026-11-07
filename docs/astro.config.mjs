// @ts-check
// The dsp-kitchen documentation: Starlight (Astro), with the look of neuro-kitchen.github.io.
// Served at https://neuro-kitchen.github.io/dsp-kitchen/ (GitHub Pages of this repository).
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightSidebarTopics from 'starlight-sidebar-topics';

const repo = 'https://github.com/neuro-kitchen/dsp-kitchen';

/**
 * An entry point: its own sidebar, opened at the folder's index page. `items` lists the pages
 * in order; by default, every page of the folder.
 */
const topic = (label, directory, icon, items = [{ autogenerate: { directory } }]) => ({
  label,
  icon,
  link: `/${directory}/`,
  items,
});

export default defineConfig({
  site: 'https://neuro-kitchen.github.io',
  base: '/dsp-kitchen',

  // Fonts as files, not data: URIs, so font-src 'self' covers them (as on the website).
  vite: { build: { assetsInlineLimit: 0 } },

  // The Content-Security-Policy is added after the build by scripts/csp.mjs (Astro's own option
  // misses Starlight's head script and style attributes).

  integrations: [
    starlight({
      title: 'dsp-kitchen',
      description: 'GPU signal processing for neural recordings: library, pipelines, spike sorting, app and server.',
      logo: { src: './src/assets/logo.svg' },
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'GitHub', href: repo }],
      editLink: { baseUrl: `${repo}/edit/main/docs/` },
      lastUpdated: true,
      routeMiddleware: './src/route-middleware.ts',
      customCss: [
        '@fontsource-variable/inter-tight',
        '@fontsource/jetbrains-mono/400.css',
        './src/styles/theme.css',
      ],
      // One entry point per purpose, each with its own sidebar (see .tasks/28).
      plugins: [
        starlightSidebarTopics([
          topic('Learn', 'learn', 'open-book', ['learn', 'learn/concepts', 'learn/python', 'learn/course']),
          topic('App', 'app', 'laptop'),
          topic('Operations', 'operations', 'puzzle', ['operations', 'operations/pipelines']),
          topic('Spike sorting', 'sorting', 'analytics'),
          topic('Reference', 'reference', 'document', [
            'reference',
            { label: 'Python API', items: [{ autogenerate: { directory: 'reference/python' } }] },
            { label: 'Rust API', items: [{ autogenerate: { directory: 'reference/rust' } }] },
            'reference/cli',
            'reference/schemas',
          ]),
          topic('Validation', 'validation', 'approve-check-circle'),
          topic('GPU lessons', 'gpu', 'rocket'),
          topic('Lab notebook', 'notebook', 'pencil'),
          topic('Deploy', 'deploy', 'server'),
          topic('Extend', 'extend', 'code-branch'),
        ]),
      ],
    }),
  ],
});
