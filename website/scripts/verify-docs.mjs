import { execSync } from 'node:child_process'
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'

const websiteDir = resolve(process.cwd())
const repoRoot = resolve(websiteDir, '..')

function fail(message) {
  console.error(`docs verify failed: ${message}`)
  process.exitCode = 1
}

function walk(dir, acc = []) {
  for (const entry of readdirSync(dir)) {
    if (
      entry === 'node_modules' ||
      entry === '.vitepress' ||
      entry === 'dist'
    ) {
      continue
    }
    const full = join(dir, entry)
    const stat = statSync(full)
    if (stat.isDirectory()) {
      walk(full, acc)
    } else {
      acc.push(full)
    }
  }
  return acc
}

function resolveDocLink(link) {
  const clean = link.replace(/#.*$/, '')
  if (!clean || clean === '/') {
    return join(websiteDir, 'index.md')
  }
  const rel = clean.replace(/^\//, '')
  return [
    join(websiteDir, `${rel}.md`),
    join(websiteDir, rel, 'index.md'),
  ].find(existsSync)
}

function readRel(relPath) {
  return readFileSync(join(websiteDir, relPath), 'utf8')
}

function routeForDoc(relPath) {
  if (relPath === 'index.md') return '/'
  if (relPath.endsWith('/index.md')) {
    return `/${relPath.replace(/\/index\.md$/u, '/')}`
  }
  return `/${relPath.replace(/\.md$/u, '')}`
}

const configSource = readFileSync(
  join(websiteDir, '.vitepress/config.ts'),
  'utf8'
)
const normalizedConfigSource = configSource.replace(/\s+/gu, ' ')
const sidebarLinks = [...configSource.matchAll(/link:\s*'([^']+)'/g)]
  .map((m) => m[1])
  .filter((link) => link.startsWith('/'))

for (const link of sidebarLinks) {
  if (!resolveDocLink(link)) {
    fail(`sidebar link does not resolve: ${link}`)
  }
}

if (!configSource.includes("hostname: 'https://utensils.io/mold/'")) {
  fail('sitemap hostname must be https://utensils.io/mold/')
}

const requiredSocialMetadata = [
  "property: 'og:title'",
  "property: 'og:description'",
  "property: 'og:image'",
  "content: 'https://utensils.io/mold/screenshots/mold-studio-desktop.png'",
  "property: 'og:url'",
  "name: 'twitter:card'",
  "content: 'summary_large_image'",
  "name: 'twitter:image'",
]
for (const metadata of requiredSocialMetadata) {
  if (!normalizedConfigSource.includes(metadata)) {
    fail(`site config missing social metadata: ${metadata}`)
  }
}

const homeSource = readRel('index.md')
if (
  !homeSource.includes('text: Local AI Image & Video Generation on Your GPU')
) {
  fail('homepage hero must describe local image and video generation')
}
if (homeSource.includes('11 Model Families')) {
  fail('homepage must not hard-code a model-family count')
}

for (const relPath of ['guide/index.md', 'models/index.md']) {
  if (readRel(relPath).toLowerCase().includes('11 model families')) {
    fail(`${relPath} must not hard-code a model-family count`)
  }
}

const expectedPackageDescription =
  'CLI-native local AI image and video generation for people, scripts, and agents'
for (const relPath of ['Cargo.toml', 'crates/mold-cli/Cargo.toml']) {
  const manifest = readFileSync(join(repoRoot, relPath), 'utf8')
  if (!manifest.includes(`description = "${expectedPackageDescription}"`)) {
    fail(`${relPath} must use the canonical package description`)
  }
}

const visibleLinks = new Set(sidebarLinks)
const requiredVisibleDocs = [
  'guide/video.md',
  'guide/iphone.md',
  'docs/catalog.md',
  'models/minimax-h3.md',
]
for (const relPath of requiredVisibleDocs) {
  const route = routeForDoc(relPath)
  if (!visibleLinks.has(route)) {
    fail(`published docs page is not linked from sidebar: ${route}`)
  }
}

const modelIndex = readRel('models/index.md')
const modelPages = readdirSync(join(websiteDir, 'models'))
  .filter((name) => name.endsWith('.md') && name !== 'index.md')
  .map((name) => `models/${name}`)
const nestedModelPages = new Map([['models/ltx-video.md', 'models/ltx2.md']])
for (const relPath of modelPages) {
  const route = routeForDoc(relPath)
  const parentPath = nestedModelPages.get(relPath)
  if (parentPath) {
    const childLink = `./${relPath.split('/').at(-1)}`
    if (!readRel(parentPath).includes(childLink)) {
      fail(`nested model guide is not linked from ${parentPath}: ${route}`)
    }
    continue
  }
  if (!visibleLinks.has(route)) {
    fail(`model guide is not linked from the sidebar: ${route}`)
  }
  if (!modelIndex.includes(`](${route})`)) {
    fail(`model guide is not linked from the models overview: ${route}`)
  }
}

for (const relPath of [
  'guide/generating.md',
  'guide/video.md',
  'guide/feature-matrix.md',
]) {
  const source = readRel(relPath)
  for (const modelRoute of ['/models/minimax-h3', '/models/wan']) {
    if (!source.includes(modelRoute)) {
      fail(`${relPath} must link to ${modelRoute}`)
    }
  }
}

const h3ModelDoc = readRel('models/minimax-h3.md')
const requiredH3DownloadFacts = [
  'minimax-h3-fl2va:comfy-pruned-int8',
  'minimax-h3-ref2va:comfy-pruned-int8',
  'minimax-h3-fl2va:comfy-pruned-int8-turbo-8step',
  'minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p',
  'minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-v1.1',
  'minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-768p',
  'minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-r21',
  'minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-r21',
  'minimax-h3-ref2va:comfy-pruned-int8-turbo-4step-r21',
  '1,956,193,000',
  '1,956,192,992',
  'dc559027db79c174125df4d827db55cd11178860',
  'lightx2v/Minimax-h3-Turbo',
  '05ef678438e84933c406131b59abbf86919b3aac',
  'drbaph/MiniMax-H3-Turbo-Lora-ComfyUI',
  'be8eb3ea3466cbb7def202ffec0d2fdc054256ac',
  '298,177,224',
  '327,035,608',
  '326,935,264',
  '44,438,283,310',
  '44,438,283,318',
  '42,780,267,542',
  '42,809,125,926',
  '42,809,025,582',
  // A rank-21 tier is an APPROXIMATION of the adapter it ships beside, and
  // the page must say so in the sentence a user reads before the pull.
  'lossy low-rank approximation',
  'capture-scope UAT override',
  '42,482,090,318',
  '63,452,470,480',
  '20,970,379,616',
  '15,687,142,551',
  '5,207,808,496',
  '605,254,808',
  '11,504,847',
  '1344x768',
  // The compact envelope became a RULE: the canvas, the clip length, and the
  // base tier's step count are ranges, and 1344x768 x 124 frames is only the
  // shape the memory bounds were MEASURED at. Pinned because the interesting
  // way this doc goes wrong is restating either as a fixed contract.
  '107 to 345 frames on the `17n+5` grid at 24 fps',
  '2 to 50 terminal-inclusive sampler grid points',
  // Metal stopped being an unimplemented backend in #1164 and, since #1323, is
  // admitted by the frozen contract and shipped in the macOS artifacts -- but
  // only a reduced-size render is retained. Pin the default-resolution
  // caveat and phase-streaming facts so disk size is not described as residency.
  'The CPU backend remains unavailable',
  // Both compact task partitions execute since #825. Pinned because the
  // interesting way this doc goes wrong is leaving the old "Ref2VA executes on
  // no released build" claim behind, or describing the ordered set as a
  // reviewed list of shapes rather than a per-request derivation.
  'Supported Ref2VA request',
  'order is authority',
  'correctness-only',
  'the default-resolution H3 Metal path remains unqualified',
  'streams Qwen language layers and DiT blocks',
  'every territory',
  'shared-server, and hosted paths',
  'model distribution or redistribution',
  'does not require a separate clickthrough',
]
for (const fact of requiredH3DownloadFacts) {
  if (!h3ModelDoc.includes(fact)) {
    fail(`MiniMax H3 model guide missing required scoped fact: ${fact}`)
  }
}

let rustEnvVars
try {
  rustEnvVars = new Set(
    execSync(
      "rg -o 'MOLD_[A-Z0-9_]+' crates --glob '!**/*test*' -g'*.rs' | sed 's/.*://' | sort -u",
      { cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] }
    )
      .trim()
      .split('\n')
      .filter(Boolean)
  )
} catch {
  console.warn(
    'warning: ripgrep (rg) not found -- skipping env-var coverage check'
  )
  rustEnvVars = new Set()
}

const ignoredEnvVars = new Set([
  'MOLD_BUILD_CHANNEL',
  'MOLD_BUILD_DATE',
  'MOLD_GIT_SHA',
  'MOLD_GIT_SHA_SHORT',
  'MOLD_VERSION',
  // Build-time only -- consumed by crates/mold-server/build.rs to stage
  // the web SPA bundle for rust-embed. Not user-facing runtime config.
  'MOLD_EMBED_WEB_DIR',
  'MOLD_WEB_DIST',
  // Build-time only -- `option_env!` paths that flake.nix bakes into the
  // Nix-built binary so Framewise upscale finds its bundled ffmpeg/ffprobe
  // (#1506). Not a runtime knob: a cargo build reads them at compile time
  // and falls back to PATH lookup.
  'MOLD_BUNDLED_FFMPEG',
  'MOLD_BUNDLED_FFPROBE',
  // Debug / dev / test-only -- intentionally not user-facing.
  // `MOLD_FLUX2_DUMP_LATENT` is a developer probe that dumps pre-VAE
  // latents to a path; `MOLD_NVFP4_PROBE_PATH` gates the NVFP4 single-file
  // load probe behind an external test fixture; `MOLD_TEST_CLIP_TOKENIZER`
  // is a unit-test fixture path. Surfacing them in docs would invite users
  // to set them in normal operation, which is wrong.
  'MOLD_FLUX2_DUMP_LATENT',
  'MOLD_DIFF_BF16',
  'MOLD_DIFF_GGUF',
  'MOLD_NVFP4_PROBE_PATH',
  // `MOLD_LTX25_GGUF_SMOKE` gates the ignored real-file GGUF smoke test
  // behind an installed checkpoint path; `MOLD_STORE_ENV_VARS` is a Rust
  // const NAME in mold-server's test_support (the hermetic-store guard),
  // not an environment variable at all — the scan's regex cannot tell.
  'MOLD_LTX25_GGUF_SMOKE',
  'MOLD_STORE_ENV_VARS',
  // Batch transaction subprocess fixtures and their stdout marker. These are
  // compiled only for Rust tests and are not supported runtime configuration.
  'MOLD_RESERVATION_TEST',
  'MOLD_TEST_BATCH_ATTEMPT_STATE',
  'MOLD_TEST_BATCH_GENERATION',
  'MOLD_TEST_BATCH_PARENT',
  'MOLD_PARENT_AUTHORITY_TEST',
  'MOLD_TEST_ARCHIVED_PARENT',
  'MOLD_TEST_GALLERY_EXPECTED_NAME',
  'MOLD_TEST_GALLERY_ACTION',
  'MOLD_TEST_GALLERY_NAME',
  'MOLD_TEST_GALLERY_OUTPUT',
  'MOLD_TEST_INCOMPLETE_PARENT_TAIL',
  'MOLD_TEST_PREDECESSOR_MODE',
  'MOLD_TEST_CLIP_TOKENIZER',
  'MOLD_TEST_GEMMA_GGUF',
  'MOLD_TEST_GEMMA_ROOT',
  'MOLD_TEST_LTX2_CHECKPOINT',
  // Opt-in PuLID/InsightFace parity fixtures: the antelopev2 weights are
  // non-commercial and never downloaded by the test, so this only points an
  // ignored parity test at an existing local directory.
  'MOLD_TEST_PULID_ASSETS',
  // Private H3 qualification/capture inputs. These are feature-gated evidence
  // authorities, not supported configuration for ordinary Mold releases.
  'MOLD_H3_AUTHORIZATION_RECORD',
  'MOLD_H3_CANONICAL_SERVER_FEATURES',
  'MOLD_H3_HOST_COMPILER_PATH',
  'MOLD_H3_HOST_COMPILER_VERSION',
  'MOLD_H3_MODEL',
  'MOLD_H3_MODELS_ROOT',
  'MOLD_H3_NATIVE_CUDA_TOOLCHAIN',
  'MOLD_H3_NVCC_PATH',
  'MOLD_H3_NVCC_VERSION',
  'MOLD_H3_PRIVATE_UAT_ROOT',
  'MOLD_H3_PRIVATE_VAE_ARTIFACT_ROOT',
  'MOLD_H3_PRIVATE_VAE_STAGING_ROOT',
  'MOLD_H3_QWEN_HEADER_SHA256',
  'MOLD_H3_QWEN_NVFP4_PATH',
  'MOLD_H3_QWEN_PATH',
  'MOLD_H3_RUNTIME_CODE_IDENTITY_SHA256',
  'MOLD_H3_RUNTIME_QUALIFICATION_RECORD',
  'MOLD_H3_STAGING_ROOT',
  // Internal desktop migration bootstrap override, not a supported end-user
  // configuration knob.
  'MOLD_HOME_POINTER_PATH',
  // Execution-plan classifier/error-display sentinels, not real settings.
  'MOLD_NOT_A_SHAPING_VARIABLE',
  'MOLD_X',
  // LTX-2.5 CUDA qualification knobs whose names the UAT harness pins ahead
  // of their emitters (crates/mold-inference/src/ltx2/provenance_vocabulary.rs,
  // #1398/#1414). The cuda-core and gguf-runtime PRs ship the real readers
  // together with their website/guide/configuration.md rows; these entries
  // may be removed once those rows are on main (the check is one-directional,
  // so leaving them is harmless).
  'MOLD_LTX2_ATTN_F32',
  'MOLD_LTX2_INT8',
  'MOLD_LTX2_QMATMUL',
])
const docsText = walk(websiteDir)
  .filter((file) => /\.(md|ts|css)$/u.test(file))
  .map((file) => readFileSync(file, 'utf8'))
  .join('\n')

for (const envVar of rustEnvVars) {
  if (ignoredEnvVars.has(envVar)) continue
  if (!docsText.includes(envVar)) {
    fail(`env var used in code but not documented in website/: ${envVar}`)
  }
}

const websiteDocs = walk(websiteDir).filter((file) => /\.md$/u.test(file))
for (const file of websiteDocs) {
  const source = readFileSync(file, 'utf8')
  if (/\bmold\s+catalog\b/u.test(source)) {
    fail(
      `stale removed catalog CLI reference found in ${file.replace(`${websiteDir}/`, '')}`
    )
  }
}

const validationSource = readFileSync(
  join(repoRoot, 'crates/mold-core/src/validation.rs'),
  'utf8'
)
const loraMatch = validationSource.match(
  /pub const LORA_CAPABLE_FAMILIES: &\[&str\] = &\[\n(?<body>[\s\S]*?)\n\];/u
)
if (!loraMatch?.groups?.body) {
  fail('could not parse LORA_CAPABLE_FAMILIES from validation.rs')
} else {
  const codeFamilies = [...loraMatch.groups.body.matchAll(/"([^"]+)"/gu)].map(
    (m) => m[1]
  )
  const matrix = readRel('guide/feature-matrix.md')
  for (const family of codeFamilies) {
    if (!matrix.includes(`| ${family}`) && !matrix.includes(`| ${family} `)) {
      fail(`LoRA-capable family missing from feature matrix: ${family}`)
    }
  }
}

const apiDocs = readRel('api/index.md')
const requiredApiEndpoints = [
  '/api/generate/reference-upload-sessions',
  '/api/generate/reference-upload',
  '/api/gallery/media-token',
  '/api/pairing/sessions',
  '/api/pairing/claim',
  '/api/gallery/preview/:name',
  '/api/downloads',
  '/api/downloads/:id',
  '/api/downloads/stream',
  '/api/catalog/families',
  '/api/catalog/search',
  '/api/catalog/installed',
  '/api/catalog/:id',
  '/api/catalog/:id/download',
  '/api/resources',
  '/api/resources/stream',
  '/api/config/model/:name/placement',
]
for (const endpoint of requiredApiEndpoints) {
  if (!apiDocs.includes(endpoint)) {
    fail(`API reference missing endpoint: ${endpoint}`)
  }
}

const requiredDocs = [
  'guide/feature-matrix.md',
  'guide/remote-workflows.md',
  'guide/performance.md',
  'guide/custom-models.md',
]

for (const relPath of requiredDocs) {
  const full = join(websiteDir, relPath)
  if (!existsSync(full)) {
    fail(`required docs page missing: ${relPath}`)
  }
}

if (!process.exitCode) {
  console.log('docs verify passed')
}
