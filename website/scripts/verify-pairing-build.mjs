import { readFileSync, existsSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')
for (const name of ['index.html', 'pair.js', 'pair.css']) {
  const source = readFileSync(resolve(root, 'public/pair', name), 'utf8')
  const output = readFileSync(
    resolve(root, '.vitepress/dist/pair', name),
    'utf8'
  )
  if (output !== source)
    throw new Error(`Pairing artifact changed trust boundary: ${name}`)
}
if (existsSync(resolve(root, '.vitepress/dist/pair.html'))) {
  throw new Error('VitePress must not emit an executable pairing page')
}
function inspect(directory) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name)
    if (entry.isDirectory()) inspect(path)
    else if (/\.(?:js|html)$/.test(entry.name)) {
      const source = readFileSync(path, 'utf8')
      if (
        /googletagmanager\.com|google-analytics\.com|G-RG6PPTGX2T|createAnalytics/.test(
          source
        )
      )
        throw new Error(
          `Third-party tracking code shares the Studio key-storage origin: ${entry.name}`
        )
      if (/<script\b[^>]*\bsrc=["'](?:https?:)?\/\//i.test(source))
        throw new Error(
          `External script shares the Studio key-storage origin: ${entry.name}`
        )
    }
  }
}
inspect(resolve(root, '.vitepress/dist'))
console.log(
  'Standalone pairing and first-party-only website artifacts verified'
)
