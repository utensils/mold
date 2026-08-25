import { PreparationRequestGuard } from "../lib/preparedExpansion";
import type { MobileBackgroundTaskLease } from "./backgroundTask";

/**
 * One generation submission's cancellable work and finite background lease.
 * The lease stays owned here until server admission takes responsibility for
 * releasing it, which makes every early return use the same cleanup path.
 */
export class MobileSubmissionAttempt {
  private backgroundTaskHandedOff = false;
  private released = false;

  constructor(
    private readonly guard: PreparationRequestGuard,
    private readonly token: number,
    private readonly backgroundTask: MobileBackgroundTaskLease,
  ) {}

  get signal(): AbortSignal {
    return this.guard.signalFor(this.token);
  }

  isCurrent(): boolean {
    return this.guard.isCurrent(this.token);
  }

  handoffBackgroundTask(): MobileBackgroundTaskLease {
    this.backgroundTaskHandedOff = true;
    return this.backgroundTask;
  }

  releaseOwnedResources(): Promise<void> {
    if (this.released) return Promise.resolve();
    this.released = true;
    return this.backgroundTaskHandedOff ? Promise.resolve() : this.backgroundTask.release();
  }
}

/** Coordinates the single generation submission that may prepare at a time. */
export class MobileSubmissionAttempts {
  constructor(private readonly guard = new PreparationRequestGuard()) {}

  begin(backgroundTask: MobileBackgroundTaskLease): MobileSubmissionAttempt {
    const token = this.guard.begin();
    return new MobileSubmissionAttempt(this.guard, token, backgroundTask);
  }

  invalidate(): void {
    this.guard.invalidate();
  }
}
