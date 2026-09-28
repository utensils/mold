---
title: Pair a phone
description: Open a mold pairing code on a phone.
sidebar: false
aside: false
outline: false
---

<script setup>
import { onMounted, ref } from 'vue'

// The pairing code rides in the fragment (#...), which the browser never
// sends to this site. It is read here, in the page, and never shown: only
// the machine's name and address, and a button that hands the code to the
// Mold app.
const machine = ref('')
const address = ref('')
const appLink = ref('')

onMounted(() => {
  const fields = new URLSearchParams(window.location.hash.slice(1))
  if (fields.get('version') === '1' && /^https?:\/\//i.test(fields.get('base_url') ?? '')) {
    machine.value = fields.get('name') ?? ''
    try {
      address.value = new URL(fields.get('base_url')).host
    } catch {
      address.value = ''
    }
    appLink.value = `mold://pair?${fields.toString()}`
  }
  // Out of the address bar, history and any screenshot of this page.
  if (window.location.hash) {
    history.replaceState(null, '', window.location.pathname + window.location.search)
  }
})
</script>

# Pair a phone

<p v-if="appLink">
  This is a pairing code for <strong>{{ machine || 'a mold machine' }}</strong>
  at <code>{{ address }}</code>.
  Your phone opened it in the browser because no app that reads it is installed.
</p>
<p v-else>
  This page is where a mold pairing code opens on a phone without the app.
  Scan the code on your computer again once the app is installed.
</p>

## iPhone or iPad

Install **[Mold Studio Companion](/guide/companion)**, then point the Camera at
the code again. Mold Studio opens, names the machine, and pairs only when you
tap **Pair**. You can also scan it inside the app: **Machines ▸ Add a Machine…
▸ Scan a Pairing Code**.

## The Mold app on iPhone or Android

The older [Mold app](/guide/iphone) reads the same code.

<template v-if="appLink">
  <p>
    Open it only if you made this code yourself: the phone will send what you
    make to <code>{{ address }}</code>.
  </p>
  <p>
    <a :href="appLink" class="pair-open">Open in Mold</a>
  </p>
</template>
<p v-else>
  In the app, open <strong>Add a machine ▸ Scan pairing code</strong>.
</p>

## Your code stays on your phone

Everything after the `#` in a pairing link stays in your browser: it is never
sent to this site. For a machine with an API key the code works once and
expires after two minutes; make a new one on your computer (**Pair a Phone…**
in Mold Studio for Mac, or **Settings ▸ Mobile pairing** in the desktop or web
app) if it has run out.

<style scoped>
.pair-open {
  display: inline-block;
  padding: 10px 20px;
  border-radius: 10px;
  background: var(--vp-button-brand-bg);
  color: var(--vp-button-brand-text) !important;
  font-weight: 600;
  text-decoration: none !important;
}
</style>
