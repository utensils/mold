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
    if (
      fields.get('version') !== '2' ||
      !/^[a-f0-9]{64}$/.test(fields.get('invite_seed') ?? '')
    )
      return
    let address
    try {
      address = new URL(fields.get('base_url'))
      if (
        !['http:', 'https:'].includes(address.protocol) ||
        address.username ||
        address.password
      )
        return
    } catch {
      return
    }
    document.getElementById('machine').textContent =
      fields.get('name') || 'your Mold machine'
    document.getElementById('address').textContent = address.host
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
