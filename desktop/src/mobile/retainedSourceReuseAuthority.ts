export interface RetainedSourceReuseSnapshot<T> {
  version: number;
  value: T;
}

/** Versioned, one-draft authority for private gallery media. */
export class RetainedSourceReuseAuthority<T> {
  private version = 0;
  private value: T | null = null;

  begin(): number {
    this.version += 1;
    this.value = null;
    return this.version;
  }

  invalidate(): void {
    this.version += 1;
    this.value = null;
  }

  setIfCurrent(version: number, value: T): boolean {
    if (version !== this.version) return false;
    this.value = value;
    return true;
  }

  snapshot(): RetainedSourceReuseSnapshot<T> | null {
    return this.value === null ? null : { version: this.version, value: this.value };
  }

  isCurrent(version: number): boolean {
    return version === this.version;
  }
}
