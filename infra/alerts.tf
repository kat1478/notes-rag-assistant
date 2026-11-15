# Cost guardrails. The account has no spend limit (advanced features → Paid plan), and reserved concurrency is not
# available: the account quota is 10 concurrent executions and AWS keeps 10 unreserved. That quota already caps the
# demo; these two alerts make overspend visible — the budget with Billing's lag, the alarm within the hour.

locals {
  alerts = var.alert_email != ""
}

resource "aws_budgets_budget" "monthly" {
  count        = local.alerts ? 1 : 0
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Actual spend at 80 % and 100 %, plus the forecast crossing 100 % — the earliest warning Budgets can give.
  dynamic "notification" {
    for_each = [
      { type = "ACTUAL", threshold = 80 },
      { type = "ACTUAL", threshold = 100 },
      { type = "FORECASTED", threshold = 100 },
    ]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value.threshold
      threshold_type             = "PERCENTAGE"
      notification_type          = notification.value.type
      subscriber_email_addresses = [var.alert_email]
    }
  }
}

resource "aws_sns_topic" "alerts" {
  count = local.alerts ? 1 : 0
  name  = "${var.name}-alerts"
}

# AWS sends a confirmation email; the alarm stays silent until the link in it is clicked.
resource "aws_sns_topic_subscription" "email" {
  count     = local.alerts ? 1 : 0
  topic_arn = aws_sns_topic.alerts[0].arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# Billing data lags by hours; invocations are a near-real-time proxy for cost (Lambda time + paid LLM calls).
resource "aws_cloudwatch_metric_alarm" "invocations" {
  count               = local.alerts && local.deploy_function ? 1 : 0
  alarm_name          = "${var.name}-invocations-spike"
  alarm_description   = "More than ${var.invocations_alarm_per_hour} demo requests in an hour — check for abuse."
  namespace           = "AWS/Lambda"
  metric_name         = "Invocations"
  dimensions          = { FunctionName = aws_lambda_function.app[0].function_name }
  statistic           = "Sum"
  period              = 3600
  evaluation_periods  = 1
  threshold           = var.invocations_alarm_per_hour
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching" # no traffic is the normal state, not an error
  alarm_actions       = [aws_sns_topic.alerts[0].arn]
  ok_actions          = [aws_sns_topic.alerts[0].arn]
}
