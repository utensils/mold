#!/usr/bin/env python3
"""Read-only Lambda relay health: aggregate metrics and current-host diagnostics only."""
import argparse
import datetime as dt
import json
import re
import subprocess
import time

READ_OPERATIONS = {
    ("lambda", "get-function-configuration"), ("lambda", "get-function-concurrency"),
    ("dynamodb", "get-item"), ("cloudwatch", "get-metric-data"),
    ("logs", "start-query"), ("logs", "get-query-results"), ("logs", "stop-query"),
}
EVENTS = ("refusal", "post gone", "connection verification", "failure")
REASONS = set("400 403 409 observed alive gone unavailable Error AbortError TimeoutError "
              "GoneException LimitExceededException ThrottlingException "
              "ProvisionedThroughputExceededException InternalServerError AccessDeniedException".split())


def aws(args, options):
    if tuple(args[:2]) not in READ_OPERATIONS:
        raise ValueError("Diagnostics permits only read operations and bounded log queries")
    result = subprocess.run(["aws", "--profile", options.profile, "--region", options.region,
                             "--no-cli-pager", *args], capture_output=True, text=True, timeout=45)
    if result.returncode:
        code = re.search(r"An error occurred \(([A-Za-z0-9_.-]+)\)", result.stderr)
        raise RuntimeError(f"AWS {args[0]} {args[1]} failed ({code[1] if code else 'Unknown'}); response omitted")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def host_key(value):
    if not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Host ID must be exactly 32 lowercase hexadecimal characters")
    return "v3-host#" + value


def host_summary(host, now):
    if not host:
        return {"present": False}
    return {"present": True, "ready": bool(host.get("connectionId")) and host.get("expiresAt", 0) > now,
            "lease_seconds_remaining": max(0, int(host.get("expiresAt", 0)) - now),
            "live_guests": sum(peer.get("expiresAt", 0) > now for peer in host.get("guests", {}).values())}


def diagnostic_query(connection_ids):
    if not connection_ids:
        return None  # Never broaden an absent host into a fleet-wide log query.
    if len(connection_ids) > 33 or any(not re.fullmatch(r"[A-Za-z0-9_+=-]{1,256}", value)
                                      for value in connection_ids):
        raise ValueError("Invalid or oversized current host connection scope")
    # Exact whitespace-delimited connection IDs only; they never leave the report.
    scope = " or ".join(f"@message like /\\s{re.escape(value)}(\\s|$)/" for value in connection_ids)
    events = "|".join(EVENTS)
    return (f"filter ({scope}) | parse @message /Mold secure relay (?<event>{events})\\s+(?<reason>[A-Za-z0-9-]+)/ "
            "| filter ispresent(event) | stats count() as count by event, reason | sort count desc | limit 100")


def log_summary(rows):
    result = []
    for row in rows:
        entry = {item["field"]: item["value"] for item in row}
        if entry.get("event") not in EVENTS or not entry.get("count", "").isdigit():
            continue
        reason = entry.get("reason")
        result.append({"event": entry["event"], "reason": reason if reason in REASONS else "other",
                       "count": int(entry["count"])})
    return result


def metric_summary(metric):
    values = metric.get("Values", [])
    result = {"metric": metric["Label"], "status": metric.get("StatusCode"), "periods": len(values)}
    if values:
        result["sum" if metric["Label"].endswith(" Sum") else "maximum_period_value"] = (
            sum(values) if metric["Label"].endswith(" Sum") else max(values))
    return result


def metric_queries(options):
    queries = []
    def add(label, namespace, name, dimensions, stat):
        queries.append({"Id": "m" + str(len(queries)), "Label": f"{label} {name} {stat}",
                        "MetricStat": {"Metric": {"Namespace": namespace, "MetricName": name,
                            "Dimensions": [{"Name": key, "Value": value} for key, value in dimensions.items()]},
                            "Period": 3600, "Stat": stat}, "ReturnData": True})
    for role in ("frontend", "router"):
        for metric, stat in (("Invocations", "Sum"), ("Errors", "Sum"), ("Throttles", "Sum"),
                             ("ConcurrentExecutions", "Maximum"), ("Duration", "p50"), ("Duration", "p95")):
            add(role, "AWS/Lambda", metric, {"FunctionName": "mold-relay-" + role}, stat)
    for metric in ("Count", "5XXError", "4XXError"):
        add("REST", "AWS/ApiGateway", metric, {"ApiName": options.http_api_name, "Stage": "production"}, "Sum")
    for metric in ("MessageCount", "ConnectCount", "ClientError", "IntegrationError", "ExecutionError"):
        add("WebSocket", "AWS/ApiGateway", metric, {"ApiId": options.websocket_api_id, "Stage": "production"}, "Sum")
    for metric in ("ConsumedReadCapacityUnits", "ConsumedWriteCapacityUnits", "ReadThrottleEvents", "WriteThrottleEvents"):
        add("table", "AWS/DynamoDB", metric, {"TableName": options.table}, "Sum")
    return queries


def collect(options):
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(hours=options.hours)
    now = int(end.timestamp())
    report = {"start": start.isoformat(), "end": end.isoformat(), "host_id": options.host_id,
              "metric_scope": "Shared relay fleet; metrics are not attributed to one host",
              "metric_period_seconds": 3600,
              "percentile_note": "Maximum hourly percentile, not a percentile across the whole window",
              "functions": {}}
    for role in ("frontend", "router"):
        name = "mold-relay-" + role
        config = aws(["lambda", "get-function-configuration", "--function-name", name,
                      "--query", "{Runtime:Runtime,MemorySize:MemorySize,Timeout:Timeout,State:State,LastUpdateStatus:LastUpdateStatus}"], options)
        concurrency = aws(["lambda", "get-function-concurrency", "--function-name", name], options)
        report["functions"][role] = {key: config[key] for key in
            ("Runtime", "MemorySize", "Timeout", "State", "LastUpdateStatus") if key in config}
        report["functions"][role]["ReservedConcurrentExecutions"] = concurrency.get("ReservedConcurrentExecutions")
    response = aws(["dynamodb", "get-item", "--table-name", options.table, "--consistent-read",
                    "--key", json.dumps({"id": {"S": host_key(options.host_id)}}),
                    "--projection-expression", "document"], options)
    host = json.loads(response.get("Item", {}).get("document", {}).get("S", "null"))
    report["host"] = host_summary(host, now)
    ids = ([host["connectionId"], *host.get("guests", {})] if host and host.get("connectionId") else [])
    query = diagnostic_query(ids)
    report["diagnostic_scope"] = "Current connection IDs only; prior disconnected sessions and unscoped frontend errors are excluded"
    report["diagnostics"] = []
    if query:
        started = aws(["logs", "start-query", "--log-group-name", "/aws/lambda/mold-relay-router",
                       "--start-time", str(int(start.timestamp())), "--end-time", str(now),
                       "--query-string", query], options)
        query_id = started["queryId"]
        deadline = time.monotonic() + 30
        while True:
            result = aws(["logs", "get-query-results", "--query-id", query_id], options)
            if result["status"] == "Complete":
                report["diagnostics"] = log_summary(result["results"])
                break
            if result["status"] not in ("Scheduled", "Running"):
                raise RuntimeError("Aggregate log query did not complete; details omitted")
            if time.monotonic() >= deadline:
                aws(["logs", "stop-query", "--query-id", query_id], options)
                raise RuntimeError("Aggregate log query exceeded its bounded wait")
            time.sleep(1)
    metrics = aws(["cloudwatch", "get-metric-data", "--start-time", start.isoformat(),
                   "--end-time", end.isoformat(), "--metric-data-queries", json.dumps(metric_queries(options))], options)
    report["metrics"] = [metric_summary(metric) for metric in metrics["MetricDataResults"]]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-id", required=True, help="Exact managed host ID (32 lowercase hex characters)")
    parser.add_argument("--hours", type=int, default=24, choices=range(1, 169), metavar="1..168")
    parser.add_argument("--profile", default="dev.urandom.io")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--table", default="mold-relay-sessions")
    parser.add_argument("--http-api-name", default="mold-relay-http")
    parser.add_argument("--websocket-api-id", default="uwijdhg05d")
    options = parser.parse_args()
    host_key(options.host_id)
    print(json.dumps(collect(options), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        # TimeoutExpired includes argv; never expose command or response payloads.
        raise SystemExit("Diagnostics AWS command timed out" if isinstance(error, subprocess.TimeoutExpired) else str(error)) from None
