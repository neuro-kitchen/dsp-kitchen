// @ts-check
// The dsp-kitchen documentation: Starlight (Astro), with the look of neuro-kitchen.github.io.
// Served at https://neuro-kitchen.github.io/dsp-kitchen/ (GitHub Pages of this repository).
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

const repo = 'https://github.com/neuro-kitchen/dsp-kitchen';

/** A sidebar group listing the pages of one folder of src/content/docs. */
const group = (label, directory) => ({ label, items: [{ autogenerate: { directory } }] });

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
      customCss: [
        '@fontsource-variable/inter-tight',
        '@fontsource/jetbrains-mono/400.css',
        './src/styles/theme.css',
      ],
      sidebar: [
        group('Get started', 'get-started'),
        group('Concepts', 'concepts'),
        {
          label: 'Guide',
          items: [
            group('Core ops', 'ops'),
            group('Pipeline', 'pipeline'),
            group('Synapse', 'synapse'),
            group('App', 'app'),
            group('CLI', 'cli'),
            group('Deploy', 'deploy'),
          ],
        },
        // Python: its own section, a guide plus the API reference generated from the stubs.
        group('Python', 'python'),
        group('Notebook', 'notebook'),
        {
          label: 'Reference',
          items: [
            { autogenerate: { directory: 'reference' } },
            { label: 'Rust (docs.rs)', link: 'https://docs.rs/dspk', attrs: { rel: 'external' } },
          ],
        },
      ],
    }),
  ],
});
