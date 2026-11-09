// @ts-check
// The dsp-kitchen documentation: Starlight (Astro), with the look of neuro-kitchen.github.io.
// Served at https://neuro-kitchen.github.io/dsp-kitchen/ (GitHub Pages of this repository).
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightSidebarTopics from 'starlight-sidebar-topics';
import course from './src/data/course.json' with { type: 'json' };

const repo = 'https://github.com/neuro-kitchen/dsp-kitchen';

/**
 * An entry point: its own sidebar, opened at the folder's index page. `items` lists the pages
 * in order; by default, every page of the folder.
 */
const topic = (label, directory, items = [{ autogenerate: { directory } }]) => ({
  label,
  link: `/${directory}/`,
  items,
});

/** A chapter of a topic: a few pages, folded unless the current page is in it. */
const chapter = (label, directory) => ({ label, collapsed: true, items: [{ autogenerate: { directory } }] });

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
      // Long code lines wrap (indentation kept) instead of scrolling sideways.
      expressiveCode: { defaultProps: { wrap: true } },
      routeMiddleware: './src/route-middleware.ts',
      customCss: [
        '@fontsource-variable/inter-tight',
        '@fontsource/jetbrains-mono/400.css',
        './src/styles/theme.css',
      ],
      // One entry point per purpose, each with its own sidebar (see .tasks/28). The topics are drawn as tabs in the header (src/components/Header.astro, grouped by
      // src/data/nav.mjs); the sidebar shows only the current topic's pages, in small chapters.
      components: {
        Header: './src/components/Header.astro',
        Sidebar: './src/components/Sidebar.astro',
      },
      plugins: [
        starlightSidebarTopics([
          topic('Learn', 'learn', [
            'learn',
            chapter('Get started', 'learn/get-started'),
            chapter('Concepts', 'learn/concepts'),
            chapter('Python', 'learn/python'),
            { label: 'Course', collapsed: true, items: ['learn/course', ...course.core.map((c) => chapter(c.label, c.directory))] },
          ]),
          topic('App', 'app'),
          topic('Operations', 'operations'),
          topic('Reference', 'reference', [
            'reference',
            chapter('Python API', 'reference/python'),
            chapter('Rust API', 'reference/rust'),
            chapter('CLI', 'reference/cli'),
            chapter('File formats', 'reference/schemas'),
          ]),
          // Plugins: each the same shape (overview, learn, reference, validation).
          topic('Synapse', 'plugins/synapse', [
            'plugins/synapse',
            {
              label: 'Learn',
              collapsed: true,
              items: [
                'plugins/synapse/learn/formats-and-probes',
                'plugins/synapse/learn/detection',
                'plugins/synapse/learn/sorting',
                'plugins/synapse/learn/sorting-in-the-app',
                'plugins/synapse/learn/curation',
                ...course.synapse.map((c) => chapter(`Course: ${c.label}`, c.directory)),
              ],
            },
            {
              label: 'Reference',
              collapsed: true,
              items: [
                chapter('Python API', 'plugins/synapse/reference/python'),
                chapter('Sorters', 'plugins/synapse/reference/sorters'),
                chapter('Sorter settings', 'plugins/synapse/reference/settings'),
              ],
            },
            chapter('Validation', 'plugins/synapse/validation'),
          ]),
          topic('Validation', 'validation'),
          topic('GPU lessons', 'gpu'),
          topic('Lab notebook', 'notebook'),
          topic('Deploy', 'deploy'),
          topic('Extend', 'extend'),
          topic('Changelog', 'changelog'),
        ]),
      ],
    }),
  ],
});
