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
