# Relay throughput and AWS cost evidence

Measured 2026-10-11 in isolated Oregon staging. These are engineering estimates
before free tiers, credits and taxes, not an AWS invoice or a production guarantee.
Production remains Lambda; no always-on compute was provisioned.

## Continuous encrypted transfer

Rust client and host used the debug UAT build, 64 outer frames, 1 MiB per HTTP
stream, and at most four 256 KiB ciphertext objects per peer/direction. Stage router
CodeSha256 was `QPmS9I4VjxoZiQa/MDHWqWkJLaTYUCORztri4sakhIU=` (source checkpoint
`a5dc8919d`). No runtime/configuration change occurred during the transfer.

| Measurement | Result |
| --- | ---: |
| Start UTC | 2026-10-11 20:46:42.416564 |
| Subprocess end UTC, derived from elapsed time | approximately 20:51:56.107 |
| Verified application response bytes | 209,715,200 (200 MiB) |
| Application duration | 313.599 s |
| Throughput | 0.638 MiB/s |
| Client maximum resident set | 25,608,192 bytes |
| Allocated host ciphertext objects | 788 |
| Allocated ciphertext object bytes | 197,985,422 |
| Original reliable frames represented by those objects | 12,201 |
| Object bytes / application bytes | 94.4% (includes framing/encryption overhead) |

This verifies uninterrupted transfer beyond the previous five-minute failure,
exact content and bounded client memory. It does not establish release-build
throughput or upload memory behavior. A separate release-build measurement is
required before attributing the remaining throughput limit to the protocol.

CloudWatch Logs Insights queried `20:46:42Z` through `20:51:58Z`: 8,212 Lambda
REPORT records, 1,105.522 billed seconds at 256 MiB, execution-duration p50
103.03 ms and p95 317.99 ms. Object counts were deduplicated by ID and direction,
restricted to fixture namespace `34750f98bd59fcfc946da45aaabe933b`. Only aggregate
counts were retained; no presigned URLs, credentials or plaintext were logged.
One admission refusal and one post-to-closed-connection diagnostic occurred in
the interval. Neither interrupted the completed transfer.

Whole-minute CloudWatch metrics from `20:46:00Z` through `20:52:00Z` recorded:
15,291 API Gateway messages, 84,902 DynamoDB consumed read units, 98,318 consumed
write units, zero Lambda errors/throttles and peak concurrency 29 of the staging
limit 32. Minute metrics include idle control traffic from another admitted test
host and the margins around the transfer; Lambda REPORT counts also cover the
whole function. These are conservative interval allocations, not exact per-user
billing attribution. The test used synthetic content and no paid GPU generation.

## Reproducible price model

Rates below were checked against AWS's public Oregon Price List on 2026-10-11.
The exact regional JSON catalogs are linked alongside the user-facing pricing
pages so the tier, operation and region can be independently checked.

| Meter | USD rate | Source |
| --- | ---: | --- |
| WebSocket message (32 KiB billing increment) | 0.000001 | [API Gateway pricing](https://aws.amazon.com/api-gateway/pricing/), [Oregon catalog](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonApiGateway/current/us-west-2/index.json) |
| WebSocket connection minute | 0.00000025 | Same API Gateway catalog |
| Lambda request | 0.0000002 | [Lambda pricing](https://aws.amazon.com/lambda/pricing/), [Oregon catalog](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSLambda/current/us-west-2/index.json) |
| Lambda ARM GB-second, first tier | 0.0000133334 | Same Lambda catalog |
| DynamoDB Standard on-demand read unit | 0.000000125 | [DynamoDB pricing](https://aws.amazon.com/dynamodb/pricing/), [Oregon catalog](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonDynamoDB/current/us-west-2/index.json) |
| DynamoDB Standard on-demand write unit | 0.000000625 | Same DynamoDB catalog |
| S3 Standard PUT | 0.000005 | [S3 pricing](https://aws.amazon.com/s3/pricing/), [Oregon catalog](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/us-west-2/index.json) |
| S3 GET or HEAD | 0.0000004 | Same S3 catalog |
| S3 Standard GB-month | 0.023 | Same S3 catalog |
| Internet egress, first paid 10 TB tier | 0.09 per GB | [AWS data-transfer Oregon catalog](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSDataTransfer/current/us-west-2/index.json) |

All current outer messages fit a single 32 KiB increment. CloudWatch's
[MessageCount definition](https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-websocket-api-logging.html)
includes messages to and from clients. Do not double that counter. The regional
S3 catalog's zero-priced intra-AWS outbound line is not the Internet-egress rate.

| Allocated cost for this interval | USD |
| --- | ---: |
| API Gateway messages | 0.015291 |
| DynamoDB reads | 0.010613 |
| DynamoDB writes | 0.061449 |
| Lambda requests | 0.001642 |
| Lambda duration (billed seconds × 0.25 GB × ARM rate) | 0.003685 |
| Estimated S3 operations: 788 × (upload PUT + tombstone PUT + HEAD + GET) | 0.008510 |
| Subtotal before egress, logs, connection minutes and storage | 0.101190 |
| Subtotal normalized to one GiB of application data | 0.518095 |

The S3 operation line is inferred from the successfully consumed object
lifecycle, not an independently metered S3 request counter; retries and late GETs
can add requests. Connection minutes, brief ciphertext storage, zero-byte
tombstones, log ingestion/retention and Internet egress remain additional costs.
Count S3 ciphertext downloads and residual inline outbound frames (including
base64 overhead) when estimating egress; upload ingress is not Internet egress.
Payloads are cleared on ACK; empty tombstones remain until lifecycle expiry to
prevent a still-valid conditional upload grant recreating cleared payloads.

DynamoDB writes dominate this sample. Each quota-row conditional write currently
rewrites up to four exact presigned grant responses, including repeated session
credentials in URL query parameters. A bounded compressed representation can
reduce charged item size without changing authority, lease or idempotence rules.
The proposed `deflate-v1` quota document preserves exact capabilities and leaves
revision/TTL outside the compressed bytes. An SDK-generated synthetic four-object
row with a 936-character session token measured 15,483 JSON bytes versus 2,423
compressed bytes: approximately 16 versus 3 billed write units including row
metadata. That is an 81% reduction for this representative row, not every database
write. Its live benefit must be measured before claiming a lower operating cost. Removing a
cloud HEAD request alone would save little money (about $0.0016/GiB at this object
count); its latency benefit and admission implications need separate review.

## Earlier baselines and limits

After deploying bounded quota compression (router
`7NuM/cdc7x3Ntp3hbROcXHuhAmjlOfqHqE7ByGDpsgc=`), the optimized Rust build verified
20,971,520 bytes from `21:11:08.837090Z` to `21:11:41.242210Z`. Application time
was 32.387 seconds (0.618 MiB/s), CPU time 0.41 seconds and maximum RSS
21,823,488 bytes. Objects still had the 256 KiB / 16-frame limit. This indicates
that cloud coordination latency, rather than debug crypto CPU, remains the
sustained throughput limit.

The exact log interval contained 80 host objects, 19,728,069 ciphertext bytes,
1,220 represented frames, 524 Lambda invocations and 70.895 billed seconds;
duration p50 was 108.49 ms and p95 326.39 ms, with no diagnostic errors. The
`21:11:00Z`–`21:12:00Z` minute contained 825 API messages, 2,856 read units,
1,705 write units, zero errors/throttles and peak Lambda concurrency 16. Four of
those read units came from the operator's bounded row-size audit; idle-host and
minute-margin caveats still apply. Using the same rates and inferred S3 lifecycle
counts gives **$0.003453 for 20 MiB, or $0.17678/GiB**, before the same additional
costs listed above. This comparison includes both the compact storage change and
the optimized client/host build, so it does not attribute every saving to compression.

The actual live two-member quota row measured 8,832 JSON bytes versus 2,958 stored
bytes (66.5% smaller). Unlike the synthetic estimate, real concurrent Lambda
workers may use different signing-session credentials, reducing cross-URL
compression. Exact capabilities and conditional-write authority were preserved.

The initial inline-only 3 MiB sample took 13.847 seconds (0.217 MiB/s), with
830 Lambda REPORTs and 48.004 billed seconds. Increasing flow-control windows
alone did not solve bulk cost. Short object samples varied substantially with
debug-build batching and concurrency, so the uninterrupted 200 MiB run above is
the primary current performance evidence. No LAN speed, release-build speed,
multi-user saturation limit, or physical-iPhone result is inferred from it.
