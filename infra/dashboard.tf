# Demo dashboard over the per-request EMF metrics the app logs (backend/app/telemetry.py) plus Lambda's own.
# Dashboards are free up to three per account; the 8 dimensionless custom metrics stay within the free ten.

locals {
  ns = "NotesRag"
  metric_widget = {
    type   = "metric"
    width  = 12
    height = 6
  }
}

resource "aws_cloudwatch_dashboard" "demo" {
  dashboard_name = var.name
  dashboard_body = jsonencode({
    widgets = [
      # Totals over the selected time range: a low-traffic demo has a few points per day, which time series
      # draw as isolated dots, so the headline numbers come first.
      {
        type = "metric", x = 0, y = 0, width = 24, height = 4
        properties = {
          title  = "Selected time range"
          region = var.region, view = "singleValue", setPeriodToTimeRange = true, stat = "Sum"
          metrics = [
            [local.ns, "LatencyMs", { stat = "SampleCount", label = "questions" }],
            [local.ns, "CostUsd", { stat = "Sum", label = "LLM cost (USD)" }],
            [local.ns, "CostUsd", { stat = "Average", label = "cost per question (USD)" }],
            [local.ns, "LatencyMs", { stat = "p50", label = "latency p50 (ms)" }],
            [local.ns, "Refused", { stat = "Average", label = "refusal rate (regex)" }],
          ]
        }
      },
      merge(local.metric_widget, { x = 0, y = 4, properties = {
        title  = "Questions and Lambda invocations (incl. page loads), per 5 min"
        region = var.region, view = "timeSeries", period = 300
        metrics = [
          [local.ns, "LatencyMs", { stat = "SampleCount", label = "/ask questions" }],
          ["AWS/Lambda", "Invocations", "FunctionName", var.name, { stat = "Sum", label = "Lambda invocations" }],
        ]
      } }),
      merge(local.metric_widget, { x = 12, y = 4, properties = {
        title  = "Latency p50 / p95 — retrieval vs LLM (ms), per 5 min"
        region = var.region, view = "timeSeries", period = 300
        metrics = [
          [local.ns, "RetrievalMs", { stat = "p50", label = "retrieval p50" }],
          [local.ns, "RetrievalMs", { stat = "p95", label = "retrieval p95" }],
          [local.ns, "GenerationMs", { stat = "p50", label = "LLM p50" }],
          [local.ns, "GenerationMs", { stat = "p95", label = "LLM p95" }],
        ]
      } }),
      merge(local.metric_widget, { x = 0, y = 10, properties = {
        title   = "LLM cost per day (USD, billed by OpenRouter)"
        region  = var.region, view = "timeSeries", period = 86400
        metrics = [[local.ns, "CostUsd", { stat = "Sum", label = "cost" }]]
      } }),
      merge(local.metric_widget, { x = 12, y = 10, properties = {
        title  = "Refusal rate (regex detector) and generation errors, per 5 min"
        region = var.region, view = "timeSeries", period = 300
        metrics = [
          [local.ns, "Refused", { stat = "Average", label = "refusal rate (regex)" }],
          [local.ns, "GenerationError", { stat = "Sum", label = "LLM errors (extractive fallback)", yAxis = "right" }],
        ]
      } }),
      {
        type = "log", x = 0, y = 16, width = 24, height = 7
        properties = {
          title  = "Last 20 questions (no text is logged — hash and length only)"
          region = var.region, view = "table"
          query  = <<-EOQ
            SOURCE '${aws_cloudwatch_log_group.app.name}'
            | filter event = "ask"
            | fields @timestamp, model, question_hash, question_chars, sources, RetrievalMs, GenerationMs, PromptTokens, CompletionTokens, CostUsd, Refused, GenerationError
            | sort @timestamp desc
            | limit 20
          EOQ
        }
      },
    ]
  })
}
