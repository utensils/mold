import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, existsSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import { createAnalytics } from '../.vitepress/theme/analytics.mjs'

test('pairing and Studio never initialize third-party analytics', () => {
  for (const path of [
    '/mold/pair',
    '/mold/pair/',
    '/mold/pair.html',
    '/mold/studio/',
    '/mold/studio/index.html',
  ]) {
    const win = {
      location: new URL(`https://utensils.io${path}#invite_seed=secret`),
      document: {
        referrer: '',
        createElement() {
          throw Error('third-party script created')
        },
      },
      dataLayer: [],
    }
    const analytics = createAnalytics(win)
    analytics.start()
    analytics.pageView()
    assert.deepEqual(win.dataLayer, [])
  }
})

test('pairing is a static CSP-protected document outside the VitePress app', () => {
  const html = readFileSync(
    new URL('../public/pair/index.html', import.meta.url),
    'utf8'
  )
  assert.equal(existsSync(new URL('../pair.md', import.meta.url)), false)
  assert.match(html, /Content-Security-Policy/)
  assert.match(html, /default-src 'none'/)
  assert.match(html, /script-src 'self'/)
  assert.match(html, /connect-src 'none'/)
  assert.match(html, /name="referrer" content="no-referrer"/)
  assert.doesNotMatch(
    html,
    /unsafe-inline|googletagmanager|vitepress|https:\/\//
  )
})

test('pairing handoff clears history before parsing, retains secret only in memory, and never injects HTML', () => {
  const source = readFileSync(
    new URL('../public/pair/pair.js', import.meta.url),
    'utf8'
  )
  const elements = new Map(
    ['machine', 'address', 'pair-ready', 'pair-missing', 'open-studio'].map(
      (id) => [
        id,
        {
          textContent: '',
          hidden: false,
          addEventListener(name, fn) {
            this[name] = fn
          },
        },
      ]
    )
  )
  const fields = new URLSearchParams({
    version: '2',
    base_url: 'http://192.168.1.4:7680',
    name: '<img src=x>',
    invite_seed: 'a'.repeat(64),
  })
  const calls = []
  const win = {
    addEventListener(name, fn) {
      this[name] = fn
    },
    location: {
      hash: `#${fields}`,
      pathname: '/mold/pair/',
      search: '?tracking=1',
      assign(url) {
        calls.push(['navigate', url])
      },
    },
    history: {
      replaceState(...args) {
        calls.push(['history', ...args])
      },
    },
  }
  win.top = win.self = win
  runInNewContext(source, {
    window: win,
    document: { getElementById: (id) => elements.get(id) },
    URL,
    URLSearchParams,
  })
  assert.deepEqual(calls, [['history', null, '', '/mold/pair/']])
  assert.equal(elements.get('machine').textContent, '<img src=x>')
  assert.equal(elements.get('address').textContent, '192.168.1.4:7680')
  assert.equal(elements.get('open-studio').href, undefined)
  assert.equal(typeof win.hashchange, 'function')
  win.location.hash =
    '#' + fields.toString().replace('name=%3Cimg+src%3Dx%3E', 'name=Second')
  win.hashchange()
  assert.equal(elements.get('machine').textContent, 'Second')
  elements.get('open-studio').click()
  assert.equal(
    calls[2][1],
    `https://utensils.io/mold/studio/#${fields.toString().replace('name=%3Cimg+src%3Dx%3E', 'name=Second')}`
  )
})

test('navigation from an analytics-loaded docs app forces a fresh security document', async () => {
  const { leaveDocsForSecurePage } =
    await import('../.vitepress/theme/secure-pages.mjs')
  const navigation = []
  const win = {
    addEventListener(name, fn) {
      this[name] = fn
    },
    location: {
      href: 'https://utensils.io/mold/guide/',
      origin: 'https://utensils.io',
      assign: (to) => navigation.push(to),
    },
  }
  for (const path of [
    '/mold/pair/#invite_seed=secret',
    '/mold/studio/#invite_seed=secret',
  ])
    assert.equal(leaveDocsForSecurePage(win, path), false)
  assert.deepEqual(navigation, [
    'https://utensils.io/mold/pair/#invite_seed=secret',
    'https://utensils.io/mold/studio/#invite_seed=secret',
  ])
  assert.equal(leaveDocsForSecurePage(win, '/mold/guide/companion'), undefined)
  assert.equal(
    leaveDocsForSecurePage(win, 'https://other.invalid/mold/pair/'),
    undefined
  )
  const theme = readFileSync(
    new URL('../.vitepress/theme/index.ts', import.meta.url),
    'utf8'
  )
  assert.match(theme, /router.onBeforeRouteChange = beforeRouteChange/)
})
