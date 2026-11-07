import { defineRouteMiddleware } from '@astrojs/starlight/route-data';

type Entry = { type: string; href?: string; entries?: Entry[] };

const hrefs = (entries: Entry[]): string[] =>
  entries.flatMap((e) => (e.type === 'group' ? hrefs(e.entries ?? []) : e.href ? [e.href] : []));

/**
 * Previous / Next stay inside the page's topic. starlight-sidebar-topics does this itself, but
 * compares paths without the site's base (/dsp-kitchen), so its check never matches here.
 * Runs after the other middleware, when the sidebar is already the topic's.
 */
export const onRequest = defineRouteMiddleware(async (context, next) => {
  await next();
  const route = context.locals.starlightRoute;
  const inTopic = new Set(hrefs(route.sidebar as Entry[]));
  if (route.pagination.prev && !inTopic.has(route.pagination.prev.href)) route.pagination.prev = undefined;
  if (route.pagination.next && !inTopic.has(route.pagination.next.href)) route.pagination.next = undefined;
});
