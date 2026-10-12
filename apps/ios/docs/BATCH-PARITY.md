# Video batching and queue cancellation

| Surface | Batch size for videos | Cancel a whole batch |
| --- | --- | --- |
| Native iOS/iPadOS | More Options → Batch, including 1 | Queue batch header, collapsed or expanded |
| Native macOS | ControlsRow → BatchControl (already present) | QueueBatchRow group action (already present) |
| Web | Composer MakeChip (already present) | Job Details → Cancel batch |
| Tauri desktop | Composer batch control (already present) | Machine queue Job Details → Cancel batch |
| Tauri iOS/Android | MobileBatchControl (already present) | Queue/Host Job Details → Cancel batch |
| CLI | `mold run --batch N` (already present) | `mold queue cancel --batch ID` (already present) |

One output disables batching. Native iOS preserves the authored count across kind changes,
matching submission behavior. Batch cancellation preserves finished Library outputs.
Native iOS/macOS use each child's supported lifecycle route (held-only for held jobs);
older hosts without cooperative cancellation leave running children alone. Web/Tauri
use the existing durable batch DELETE after rechecking instance, batch, client and child
identity. A changed route, credential, mode or selected job invalidates confirmation.

Regression coverage: QueueStoreTests, PopulatedGenerationTests video options and collapsed
batch tests, Studio queueBatchCancel and QueueBatchCancelControl tests. Local Simulator
execution is blocked until the installed Xcode license is accepted; no physical-device
validation has been claimed.
