- **Desktop Library organization.** The Library header is now **Prints |
  Collections | Trash**: favorites, tags, and manual collections (merged
  across every connected machine), a real trash with per-tile purge
  countdowns, Restore / Delete forever, and Empty trash behind a plain confirm,
  with titles, ♥, tags, and collection membership edited in the lightbox
  aside. Deleting a print moves it to that machine's trash instead of erasing
  it; with the local server off, This device's delete, trash list, restore,
  and delete-forever still work through the native `.trash/` path. A second
  print deleted under an already-trashed filename is kept alongside the first
  (renamed `name-2.ext`) — previously trashed bytes are never overwritten —
  and Trash-view media, Save, Reveal, and Copy file path always read the
  trashed copy even when a newer live file reuses its name.
- **Print titles from Create.** The Create header's "Untitled print" is
  editable; the name rides every sibling of that print as
  `GenerateRequest.title`, is restored by Reuse settings, and becomes the
  suggested filename when you save or export.
- **Trash retention settings.** Settings ▸ Library sets how long this device
  keeps deleted prints (`gallery.trash_retention_days`: 1 day … 1 year, or
  Forever); Machines ▸ machine ▸ Storage sets each remote machine's own
  retention and shows its trash count with an Empty trash action.
