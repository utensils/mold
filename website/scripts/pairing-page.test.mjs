import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, existsSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
test('every Mold document excludes third-party analytics from the shared key-storage origin', () => {
  const theme = readFileSync(
    new URL('../.vitepress/theme/index.ts', import.meta.url),
    'utf8'
  )
  assert.doesNotMatch(
    theme,
    /createAnalytics|analytics\.mjs|gtag|googletagmanager/
  )
  assert.equal(
    existsSync(new URL('../.vitepress/theme/analytics.mjs', import.meta.url)),
    false
  )
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
    [
      'machine',
      'address',
      'pair-ready',
      'pair-missing',
      'open-studio',
      'open-native',
    ].map((id) => [
      id,
      {
        textContent: '',
        hidden: false,
        addEventListener(name, fn) {
          this[name] = fn
        },
      },
    ])
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

test('navigation from the docs app forces a fresh security document', async () => {
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

test('Regular handoff names its mode and rejects mixed or proxy invitations', () => {
  const source = readFileSync(
    new URL('../public/pair/pair.js', import.meta.url),
    'utf8'
  )
  function inspect(extra = {}) {
    const elements = new Map(
      [
        'machine',
        'address',
        'mode',
        'pair-ready',
        'pair-missing',
        'open-studio',
        'open-native',
      ].map((id) => [
        id,
        {
          hidden: false,
          textContent: '',
          addEventListener(name, fn) {
            this[name] = fn
          },
        },
      ])
    )
    const fields = new URLSearchParams({
      version: '2',
      mode: 'regular',
      base_url: 'http://192.168.1.4:7680',
      instance_id: 'host',
      name: 'Studio',
      pairing_token: 'a'.repeat(64),
      expires_at: String(Math.floor(Date.now() / 1000) + 120),
      ...extra,
    })
    const win = {
      location: { hash: `#${fields}`, pathname: '/mold/pair/', assign() {} },
      history: { replaceState() {} },
      addEventListener() {},
    }
    win.top = win.self = win
    runInNewContext(source, {
      window: win,
      document: { getElementById: (id) => elements.get(id) },
      URL,
      URLSearchParams,
    })
    return elements
  }
  const valid = inspect()
  assert.equal(valid.get('pair-ready').hidden, false)
  assert.match(valid.get('mode').textContent, /Regular.*direct/)
  assert.equal(
    inspect({ invite_seed: 'b'.repeat(64) }).get('pair-ready').hidden,
    true
  )
  assert.equal(
    inspect({ base_url: 'https://mold-link.urandom.io' }).get('pair-ready')
      .hidden,
    true
  )
  assert.equal(inspect({ expires_at: '1' }).get('pair-ready').hidden, true)
})

test('native handoff keeps invitation private and preserves browser fallback', () => {
  const source = readFileSync(
    new URL('../public/pair/pair.js', import.meta.url),
    'utf8'
  )
  const elements = new Map(
    [
      'machine',
      'address',
      'mode',
      'pair-ready',
      'pair-missing',
      'open-studio',
      'open-native',
    ].map((id) => [
      id,
      {
        hidden: false,
        textContent: '',
        addEventListener(name, fn) {
          this[name] = fn
        },
      },
    ])
  )
  const fields = new URLSearchParams({
    version: '2',
    base_url: 'http://192.168.1.4:7680',
    invite_seed: 'a'.repeat(64),
  })
  const navigation = []
  const win = {
    location: {
      hash: `#${fields}`,
      pathname: '/mold/pair/',
      assign(url) {
        navigation.push(url)
      },
    },
    history: { replaceState() {} },
    addEventListener() {},
  }
  win.top = win.self = win
  runInNewContext(source, {
    window: win,
    document: { getElementById: (id) => elements.get(id) },
    URL,
    URLSearchParams,
  })
  assert.equal(elements.get('open-native').href, undefined)
  elements.get('open-native').click()
  assert.deepEqual(navigation, [`moldstudio://pair#${fields}`])
  // If the native app is not installed, the user can still choose browser Studio.
  elements.get('open-studio').click()
  assert.equal(navigation[1], `https://utensils.io/mold/studio/#${fields}`)
  elements.get('open-native').click()
  assert.equal(navigation.length, 2)
  const html = readFileSync(
    new URL('../public/pair/index.html', import.meta.url),
    'utf8'
  )
  assert.match(html, /id="open-native"[^>]*>\s*Open iPhone or iPad app/)
  assert.match(html, /Open browser Studio/)
})
