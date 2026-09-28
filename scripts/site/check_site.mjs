// Automated checks for the landing page in _site/ (build it first with
// build_site.sh). Serves _site/ on a local port, drives Chromium with
// Playwright and exits non-zero if any check fails. Screenshots are written
// to build/TestingOutput/ as a record; no check depends on them.
//
//   cd scripts/site && npm ci && npm run check
//
// External targets (github.io, github.com) are answered with request
// interception, so the checks need no network. The IONOS frameset is served
// from http://localhost:<port>/ while the site is loaded from
// http://127.0.0.1:<port>/, so the frame is cross-origin as on klartraum.ai.

/**
 * TESTS:
 * - media: every image and video poster loads, every video is served, and videos play where the browser decodes H.264
 * - noHorizontalScroll: no horizontal page scroll at 375 px and 1280 px width
 * - linkTargets: external links use target="_top" or target="_blank"
 * - productLinks: both product cards exist with GitHub and documentation links
 * - darkMode: prefers-color-scheme: dark switches the body background to the dark token
 * - reducedMotion: prefers-reduced-motion: reduce creates no animated glyphs and keeps the videos paused
 * - frameBreakout: inside a frameset like the one on klartraum.ai, a documentation link navigates the top window
 **/

import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const siteDir = join(repoRoot, "_site");
const outDir = join(repoRoot, "build", "TestingOutput");

const EXPECTED_LINKS = {
    engine: {
        docs: "https://fortmeier.github.io/klartraum/docs/",
        github: "https://github.com/fortmeier/klartraum",
    },
    studio: {
        docs: "https://fortmeier.github.io/klartraum-studio/",
        github: "https://github.com/fortmeier/klartraum-studio",
    },
};

const DARK_BG = "rgb(17, 17, 22)";

const MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css",
    ".js": "text/javascript",
    ".jpg": "image/jpeg",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".svg": "image/svg+xml",
};

const FRAMESET_PATH = "/__klartraum_frameset.html";

function serveSite() {
    const server = createServer(async (req, res) => {
        if (req.url === FRAMESET_PATH) {
            const site = `http://127.0.0.1:${server.address().port}/`;
            res.writeHead(200, { "content-type": MIME[".html"] });
            res.end(`<!DOCTYPE html><html><head><title>klartraum.ai</title></head>
                     <frameset rows="100%"><frame src="${site}" frameborder="0"></frameset></html>`);
            return;
        }
        let path = decodeURIComponent(new URL(req.url, "http://x").pathname);
        if (path.endsWith("/")) path += "index.html";
        const file = normalize(join(siteDir, path));
        if (!file.startsWith(siteDir)) {
            res.writeHead(403).end();
            return;
        }
        try {
            const body = await readFile(file);
            res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
            res.end(body);
        } catch {
            res.writeHead(404).end();
        }
    });
    return new Promise((ok) => server.listen(0, "127.0.0.1", () => ok(server)));
}

const results = [];

async function check(name, fn) {
    try {
        await fn();
        results.push({ name, ok: true });
        console.log(`PASS  ${name}`);
    } catch (err) {
        results.push({ name, ok: false });
        console.log(`FAIL  ${name}: ${err.message}`);
    }
}

function assert(condition, message) {
    if (!condition) throw new Error(message);
}

const server = await serveSite();
const base = `http://127.0.0.1:${server.address().port}/`;
await mkdir(outDir, { recursive: true });
const browser = await chromium.launch();

// Answer every external request with a stub page, so link targets can be
// followed without network access.
async function stubExternal(context) {
    await context.route(/^https:\/\/(fortmeier\.github\.io|github\.com)\//, (route) =>
        route.fulfill({ contentType: "text/html", body: "<title>stub</title>" }));
}

try {
    await check("media", async () => {
        const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
        await page.goto(base, { waitUntil: "load" });
        const broken = await page.$$eval("img", (imgs) =>
            imgs.filter((i) => !i.complete || i.naturalWidth === 0).map((i) => i.getAttribute("src")));
        assert(broken.length === 0, `images failed to load: ${broken.join(", ")}`);
        const videos = await page.$$eval("video", (vs) =>
            vs.map((v) => ({ src: v.getAttribute("src"), poster: v.getAttribute("poster") })));
        assert(videos.length > 0, "no videos on the page");
        for (const { src, poster } of videos) {
            assert(poster, `video ${src} has no poster`);
            for (const url of [src, poster]) {
                const response = await page.request.get(new URL(url, base).href);
                assert(response.ok(), `${url} is not served (${response.status()})`);
            }
            const posterLoads = await page.evaluate((url) => new Promise((ok) => {
                const img = new Image();
                img.onload = () => ok(img.naturalWidth > 0);
                img.onerror = () => ok(false);
                img.src = url;
            }), poster);
            assert(posterLoads, `poster ${poster} does not load`);
        }
        // The open-source Chromium on Linux cannot decode H.264; there only the
        // checks above apply.
        const decodes = await page.evaluate(() =>
            document.createElement("video").canPlayType('video/mp4; codecs="avc1.64001E"') !== "");
        if (decodes) {
            await page.waitForFunction(() =>
                [...document.querySelectorAll("video")].every((v) => !v.paused && v.currentTime > 0.2),
                null, { timeout: 10000 });
        } else {
            console.log("      (this browser cannot decode H.264; video playback not checked)");
        }
        await page.close();
    });

    await check("noHorizontalScroll", async () => {
        for (const width of [375, 1280]) {
            const page = await browser.newPage({ viewport: { width, height: 800 } });
            await page.goto(base, { waitUntil: "load" });
            const { scrollWidth, clientWidth } = await page.evaluate(() => ({
                scrollWidth: document.documentElement.scrollWidth,
                clientWidth: document.documentElement.clientWidth,
            }));
            await page.screenshot({ path: join(outDir, `site-landing-${width}.png`), fullPage: true });
            assert(scrollWidth <= clientWidth, `${width}px: scrollWidth ${scrollWidth} > clientWidth ${clientWidth}`);
            await page.close();
        }
    });

    await check("linkTargets", async () => {
        const page = await browser.newPage();
        await page.goto(base, { waitUntil: "load" });
        const bad = await page.$$eval("a[href]", (links) =>
            links
                .filter((a) => /^https?:/.test(a.getAttribute("href")))
                .filter((a) => !["_top", "_blank"].includes(a.getAttribute("target")))
                .map((a) => a.getAttribute("href")));
        assert(bad.length === 0, `external links without _top/_blank: ${bad.join(", ")}`);
        await page.close();
    });

    await check("productLinks", async () => {
        const page = await browser.newPage();
        await page.goto(base, { waitUntil: "load" });
        for (const [id, links] of Object.entries(EXPECTED_LINKS)) {
            const hrefs = await page.$$eval(`#${id} a`, (as) => as.map((a) => a.getAttribute("href")));
            assert(hrefs.length > 0, `card #${id} is missing`);
            for (const [kind, url] of Object.entries(links)) {
                assert(hrefs.includes(url), `card #${id} has no ${kind} link to ${url}`);
            }
        }
        await page.close();
    });

    await check("darkMode", async () => {
        const page = await browser.newPage({ colorScheme: "dark" });
        await page.goto(base, { waitUntil: "load" });
        const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
        await page.screenshot({ path: join(outDir, "site-landing-dark.png") });
        assert(bg === DARK_BG, `dark background is ${bg}, expected ${DARK_BG}`);
        await page.close();
    });

    await check("reducedMotion", async () => {
        const page = await browser.newPage({ reducedMotion: "reduce" });
        await page.goto(base, { waitUntil: "load" });
        const animated = await page.$$eval(".rain span", (spans) =>
            spans.filter((s) => getComputedStyle(s).animationName !== "none" &&
                                s.getBoundingClientRect().width > 0).length);
        assert(animated === 0, `${animated} glyphs are animated with reduced motion`);
        await page.waitForTimeout(1000);
        const playing = await page.$$eval("video", (vs) =>
            vs.filter((v) => !v.paused || v.currentTime > 0).map((v) => v.getAttribute("src")));
        assert(playing.length === 0, `videos play with reduced motion: ${playing.join(", ")}`);
        // Without the preference, the rain is there; otherwise this check proves nothing.
        const normal = await browser.newPage({ reducedMotion: "no-preference" });
        await normal.goto(base, { waitUntil: "load" });
        const count = await normal.$$eval(".rain span", (spans) => spans.length);
        assert(count > 0, "no glyphs without reduced motion either; check is not meaningful");
        await normal.close();
        await page.close();
    });

    await check("frameBreakout", async () => {
        // Mirrors the frameset that the IONOS server delivers for klartraum.ai.
        const context = await browser.newContext();
        await stubExternal(context);
        const page = await context.newPage();
        await page.goto(`http://localhost:${server.address().port}${FRAMESET_PATH}`, { waitUntil: "load" });
        // The frame can commit its navigation after the frameset's load event.
        let frame;
        for (let i = 0; i < 50 && !frame; i++) {
            frame = page.frames().find((f) => f.url() === base);
            if (!frame) await page.waitForTimeout(100);
        }
        assert(frame, `landing page did not load inside the frameset (frames: ${page.frames().map((f) => f.url()).join(", ")})`);
        await frame.waitForSelector(`#engine a[href="${EXPECTED_LINKS.engine.docs}"]`);
        const target = EXPECTED_LINKS.engine.docs;
        await Promise.all([
            page.waitForURL(target, { timeout: 5000 }),
            frame.click(`#engine a[href="${target}"]`),
        ]);
        assert(page.url() === target, `top window is at ${page.url()}, expected ${target}`);
        await context.close();
    });
} finally {
    await browser.close();
    server.close();
}

const failed = results.filter((r) => !r.ok).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed === 0 ? 0 : 1);
