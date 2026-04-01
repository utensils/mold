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

const configSource = readFileSync(
  join(websiteDir, '.vitepress/config.ts'),
  'utf8'
)
const sidebarLinks = [...configSource.matchAll(/link:\s*'([^']+)'/g)]
  .map((m) => m[1])
  .filter((link) => link.startsWith('/'))

for (const link of sidebarLinks) {
  if (!resolveDocLink(link)) {
    fail(`sidebar link does not resolve: ${link}`)
  }
}

const rustEnvVars = new Set(
  execSync(
    "rg -o 'MOLD_[A-Z0-9_]+' crates --glob '!**/*test*' -g'*.rs' | sed 's/.*://' | sort -u",
    { cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] }
  )
    .trim()
    .split('\n')
    .filter(Boolean)
)

const ignoredEnvVars = new Set([
  'MOLD_BUILD_DATE',
  'MOLD_GIT_SHA',
  'MOLD_VERSION',
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
