---
title: Pair a device
description: Pair a Mold app or browser with an end-to-end encrypted host.
sidebar: false
aside: false
outline: false
---

<script setup>
import { onMounted, ref } from 'vue'

// The pairing code rides in the fragment (#...), which the browser never
// sends to this site. It is read here, in the page, and never shown: only the machine's name and address and an explicit handoff button.
const machine = ref('')
const address = ref('')
const appLink = ref('')
const studioLink = ref('')

onMounted(() => {
  const fields = new URLSearchParams(window.location.hash.slice(1))
  if (fields.get('version') === '2' && /^https?:\/\//i.test(fields.get('base_url') ?? '')) {
    machine.value = fields.get('name') ?? ''
    try {
      address.value = new URL(fields.get('base_url')).host
    } catch {
      address.value = ''
    }
    appLink.value = `https://utensils.io/mold/pair#${fields.toString()}`
    studioLink.value = `https://utensils.io/mold/studio/#${fields.toString()}`
  }
  // Out of the address bar, history and any screenshot of this page.
  if (window.location.hash) {
    history.replaceState(null, '', window.location.pathname + window.location.search)
  }
})
</script>

# Pair a device

<p v-if="appLink">
  This is a pairing code for <strong>{{ machine || 'a mold machine' }}</strong>
  at <code>{{ address }}</code>.
  Open it in an installed Mold app, or pair this browser with the secure Studio.
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

The [Mold app](/guide/iphone) reads the same encrypted pairing code. Update all clients before pairing.

<template v-if="appLink">
  <p>
    Open it only if you made this code yourself: the phone will send what you
    make to <code>{{ address }}</code>.
  </p>
  <p>
    <a :href="studioLink" class="pair-open">Open secure Studio</a>
  </p>
</template>
<p v-else>
  In the app, open <strong>Add a machine ▸ Scan pairing code</strong>.
</p>

## Your code stays on your device

Everything after the `#` in a pairing link stays in your browser: it is never
sent to this site. Each host-signed invitation works once and
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
