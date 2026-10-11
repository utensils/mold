let invitation = "";

/** Capture before hash history initializes; a pairing secret is never a route. */
export function captureInitialPairingLink(): void {
  if (import.meta.env.VITE_STANDALONE_STUDIO !== "1") return;
  const fragment = location.hash;
  if (!fragment.includes("invite_seed=")) return;
  invitation = `https://utensils.io/mold/pair/${fragment}`;
  history.replaceState(
    history.state,
    "",
    `${location.pathname}${location.search}#/create`,
  );
}

/** Only the access screen consumes the in-memory invitation, exactly once. */
export function takeInitialPairingLink(): string {
  const value = invitation;
  invitation = "";
  return value;
}
