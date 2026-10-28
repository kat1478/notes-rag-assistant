data "aws_caller_identity" "current" {}

locals {
  key_parameter_arn = "arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter${var.openrouter_key_parameter}"
  # The function is created only once an image exists; the first deploy step creates just the repository.
  deploy_function = var.image_digest != ""
}

resource "aws_ecr_repository" "app" {
  name                 = var.name
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true # demo: `terraform destroy` should not get stuck on old images
}

# Each deploy pushes a ~2 GB image (layers are shared between versions); keeping 3 bounds storage at a few GB.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 3 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 3 }
      action       = { type = "expire" }
    }]
  })
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/aws/lambda/${var.name}"
  retention_in_days = 14
}

resource "aws_iam_role" "lambda" {
  name = "${var.name}-lambda"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

# Least privilege: write its own logs and read one parameter. Decryption uses the AWS-managed aws/ssm key,
# whose key policy already allows callers in this account via SSM.
resource "aws_iam_role_policy" "lambda" {
  name = "logs-and-key"
  role = aws_iam_role.lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "${aws_cloudwatch_log_group.app.arn}:*" },
      { Effect = "Allow", Action = "ssm:GetParameter", Resource = local.key_parameter_arn },
    ]
  })
}

resource "aws_lambda_function" "app" {
  count         = local.deploy_function ? 1 : 0
  function_name = var.name
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}@${var.image_digest}"
  architectures = ["x86_64"]
  memory_size   = var.memory_mb
  timeout       = 60 # cold start (~25 s) + LLM call

  environment {
    variables = {
      OPENROUTER_KEY_PARAMETER = var.openrouter_key_parameter
      ASK_RATE_LIMIT           = tostring(var.ask_rate_limit)
    }
  }

  depends_on = [aws_cloudwatch_log_group.app, aws_iam_role_policy.lambda]
}

resource "aws_lambda_function_url" "app" {
  count              = local.deploy_function ? 1 : 0
  function_name      = aws_lambda_function.app[0].function_name
  authorization_type = "NONE" # public demo; abuse is capped by rate limit + the key's spend limit
}

# A public URL needs both statements since Oct 2025: InvokeFunctionUrl, and InvokeFunction restricted to URL calls.
resource "aws_lambda_permission" "url" {
  count                  = local.deploy_function ? 1 : 0
  statement_id           = "PublicFunctionUrl"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.app[0].function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

resource "aws_lambda_permission" "url_invoke" {
  count                    = local.deploy_function ? 1 : 0
  statement_id             = "PublicFunctionUrlInvoke"
  action                   = "lambda:InvokeFunction"
  function_name            = aws_lambda_function.app[0].function_name
  principal                = "*"
  invoked_via_function_url = true
}
