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

const h3ModelDoc = readRel('models/minimax-h3.md')
const requiredH3DownloadFacts = [
  'minimax-h3-fl2va:comfy-pruned-int8',
  'minimax-h3-ref2va:comfy-pruned-int8',
  'minimax-h3-fl2va:comfy-pruned-int8-turbo-8step',
  'minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p',
  '1,956,193,000',
  '1,956,192,992',
  'dc559027db79c174125df4d827db55cd11178860',
  'capture-scope UAT override',
  '42,482,090,318',
  '63,452,470,480',
  '20,970,379,616',
  '15,687,142,551',
  '5,207,808,496',
  '605,254,808',
  '11,504,847',
  '1344x768',
  'exactly 124 frames at 24 fps',
  '21 terminal-inclusive sampler grid points',
  // Metal stopped being an unimplemented backend in #1164 and became a
  // compiled correctness-only path that is still not a runnable route.
  // All three strings are pinned because the interesting way this doc goes
  // wrong is describing Metal as runnable, or quietly dropping the caveat.
  'Ref2VA execution and the CPU backend remain unavailable',
  'correctness-only',
  'Metal is not yet a runnable H3 route',
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
    'warning: ripgrep (rg) not found — skipping env-var coverage check'
  )
  rustEnvVars = new Set()
}

const ignoredEnvVars = new Set([
  'MOLD_BUILD_DATE',
  'MOLD_GIT_SHA',
  'MOLD_GIT_SHA_SHORT',
  'MOLD_VERSION',
  // Build-time only — consumed by crates/mold-server/build.rs to stage
  // the web SPA bundle for rust-embed. Not user-facing runtime config.
  'MOLD_EMBED_WEB_DIR',
  'MOLD_WEB_DIST',
  // Debug / dev / test-only — intentionally not user-facing.
  // `MOLD_FLUX2_DUMP_LATENT` is a developer probe that dumps pre-VAE
  // latents to a path; `MOLD_NVFP4_PROBE_PATH` gates the NVFP4 single-file
  // load probe behind an external test fixture; `MOLD_TEST_CLIP_TOKENIZER`
  // is a unit-test fixture path. Surfacing them in docs would invite users
  // to set them in normal operation, which is wrong.
  'MOLD_FLUX2_DUMP_LATENT',
  'MOLD_DIFF_BF16',
  'MOLD_DIFF_GGUF',
  'MOLD_NVFP4_PROBE_PATH',
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
