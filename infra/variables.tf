variable "region" {
  type    = string
  default = "eu-north-1" # fixed for the AWS Settings project
}

variable "name" {
  type    = string
  default = "notes-rag"
}

variable "image_digest" {
  description = "sha256 digest of the image pushed to ECR (set by scripts/deploy.sh). Empty = create only the repository."
  type        = string
  default     = ""
}

variable "openrouter_key_parameter" {
  description = "SSM SecureString holding the demo OpenRouter key (created by hand, never by Terraform)."
  type        = string
  default     = "/notes-rag/openrouter-api-key"
}

variable "memory_mb" {
  description = "Lambda memory; CPU scales with it, which shortens the cold start."
  type        = number
  default     = 3008
}

variable "ask_rate_limit" {
  description = "Max /ask requests per client IP per hour, per Lambda instance."
  type        = number
  default     = 10
}

variable "alert_email" {
  description = "Where cost and traffic alerts go. Set in the git-ignored infra/terraform.tfvars; empty = no alerts."
  type        = string
  default     = ""
}

variable "monthly_budget_usd" {
  description = "Monthly AWS cost budget; the demo normally costs ~$0, so any real spend is a signal."
  type        = number
  default     = 5
}

variable "invocations_alarm_per_hour" {
  description = "Lambda invocations per hour that trigger the traffic alarm (normal use is a handful)."
  type        = number
  default     = 200
}
