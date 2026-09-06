import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  createAnalytics,
  measurementId,
} from '../.vitepress/theme/analytics.mjs'

function fixture(url = 'https://utensils.io/mold/') {
  const events = [],
    scripts = [],
    stored = new Map()
  const win = {
    location: new URL(url),
    localStorage: {
      getItem: (k) => stored.get(k),
      setItem: (k, v) => stored.set(k, v),
    },
    document: {
      title: 'mold',
      referrer: 'https://example.com/?private=value',
      createElement: () => ({}),
      head: { appendChild: (s) => scripts.push(s) },
    },
    dataLayer: events,
  }
  return { win, events, scripts, stored, analytics: createAnalytics(win) }
}

test('no Google script or events before consent, including rejection', () => {
  const f = fixture()
  f.analytics.start()
  f.analytics.pageView()
  f.analytics.choose(false)
  assert.equal(f.scripts.length, 0)
  assert.equal(f.events.length, 0)
  assert.equal(f.analytics.choice(), 'denied')
})

test('accept loads once and records one sanitized page view per navigation', () => {
  const f = fixture('https://utensils.io/mold/?secret=value#fragment')
  f.analytics.choose(true)
  f.analytics.start()
  assert.equal(f.scripts.length, 1)
  assert.match(f.scripts[0].src, new RegExp(measurementId))
  const events = f.events.map((e) => Array.from(e))
  assert.equal(events[0][0], 'consent')
  assert.equal(events[0][2].ad_storage, 'denied')
  const config = events.find((e) => e[0] === 'config')[2]
  assert.equal(config.send_page_view, false)
  assert.equal(config.allow_google_signals, false)
  assert.equal(config.cookie_path, '/mold/')
  let views = events.filter((e) => e[1] === 'page_view')
  assert.equal(views.length, 1)
  assert.equal(views[0][2].page_location, 'https://utensils.io/mold/')
  assert.equal(views[0][2].page_referrer, 'https://example.com/')
  f.win.location = new URL('https://utensils.io/mold/guide/?secret=value')
  f.win.document.title = 'Guide | mold'
  f.analytics.pageView()
  views = f.events.map((e) => Array.from(e)).filter((e) => e[1] === 'page_view')
  assert.equal(views.length, 2)
  assert.equal(views[1][2].page_title, 'Guide | mold')
  assert.equal(views[1][2].page_referrer, 'https://utensils.io/mold/')
})

test('saved acceptance starts tracking; withdrawal disables subsequent events', () => {
  const f = fixture()
  f.stored.set('mold.website.analytics', 'granted')
  f.analytics.start()
  assert.equal(f.scripts.length, 1)
  f.analytics.choose(false)
  assert.equal(f.win[`ga-disable-${measurementId}`], true)
  const count = f.events.length
  f.analytics.pageView()
  assert.equal(f.events.length, count)
})

test('development, previews, and other utensils sites never load analytics', () => {
  for (const url of [
    'http://localhost:5173/mold/',
    'https://utensils.io/other/',
    'https://example.com/mold/',
  ]) {
    const f = fixture(url)
    f.analytics.choose(true)
    f.analytics.pageView()
    assert.equal(f.scripts.length, 0)
    assert.equal(f.events.length, 0)
  }
})

test('unavailable storage does not break the website or consent choice', () => {
  const f = fixture()
  f.win.localStorage = {
    getItem() {
      throw Error('blocked')
    },
    setItem() {
      throw Error('blocked')
    },
  }
  assert.doesNotThrow(() => f.analytics.start())
  f.analytics.choose(true)
  assert.equal(f.analytics.choice(), 'granted')
  assert.equal(f.scripts.length, 1)
})
