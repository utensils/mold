;(() => {
  let invitation = ''
  function consume() {
    const fragment = window.location.hash
    // Remove private material before parsing or displaying any untrusted fields.
    window.history.replaceState(null, '', window.location.pathname)
    invitation = ''
    document.getElementById('pair-ready').hidden = true
    document.getElementById('pair-missing').hidden = false
    document.getElementById('machine').textContent = ''
    document.getElementById('address').textContent = ''
    if (window.top !== window.self || fragment.length > 32768) return
    const fields = new URLSearchParams(fragment.slice(1))
    const regular = fields.get('mode') === 'regular'
    if (fields.get('version') !== '2') return
    if (regular) {
      if (
        !/^[a-f0-9]{64}$/.test(fields.get('pairing_token') ?? '') ||
        !fields.get('instance_id') ||
        !Number.isSafeInteger(Number(fields.get('expires_at'))) ||
        Number(fields.get('expires_at')) <= Date.now() / 1000 ||
        [
          'namespace',
          'host_key',
          'host_noise_key',
          'host_binding_signature',
          'websocket_url',
          'invite_seed',
          'invite',
          'endpoints',
        ].some((key) => fields.has(key))
      )
        return
    } else if (
      (fields.has('mode') && fields.get('mode') !== 'secure') ||
      fields.has('pairing_token') ||
      !/^[a-f0-9]{64}$/.test(fields.get('invite_seed') ?? '')
    )
      return
    let address
    try {
      address = new URL(fields.get('base_url'))
      if (
        !['http:', 'https:'].includes(address.protocol) ||
        address.username ||
        address.password ||
        address.pathname !== '/' ||
        address.search ||
        address.hash ||
        (regular &&
          (address.hostname === 'mold-link.urandom.io' ||
            address.hostname.endsWith('.mold-link.urandom.io')))
      )
        return
    } catch {
      return
    }
    document.getElementById('machine').textContent =
      fields.get('name') || 'your Mold machine'
    document.getElementById('address').textContent = address.host
    if (document.getElementById('mode'))
      document.getElementById('mode').textContent = regular
        ? 'Regular · direct HTTP or HTTPS'
        : 'Secure · end-to-end encrypted'
    document.getElementById('pair-ready').hidden = false
    document.getElementById('pair-missing').hidden = true
    invitation = fields.toString()
  }
  // Keep the invitation in a closure, never an href or other DOM attribute.
  document.getElementById('open-studio').addEventListener('click', () => {
    if (!invitation) return
    const target = `https://utensils.io/mold/studio/#${invitation}`
    invitation = ''
    window.location.assign(target)
  })
  window.addEventListener('hashchange', consume)
  consume()
})()
