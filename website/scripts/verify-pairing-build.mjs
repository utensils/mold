import { readFileSync, existsSync } from 'node:fs'
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
console.log('Standalone pairing artifacts verified')
