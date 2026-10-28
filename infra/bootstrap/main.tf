# One-time setup: the S3 bucket that holds Terraform state for the app stack in ../
# Its own state stays local (terraform.tfstate here, git-ignored) — it manages a single bucket.
terraform {
  required_version = ">= 1.10" # native S3 state locking (use_lockfile)
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "notes-rag", ManagedBy = "terraform" }
  }
}

variable "region" {
  type    = string
  default = "eu-north-1" # fixed for the AWS Settings project, see README
}

data "aws_caller_identity" "current" {}

# Bucket names are global; the account ID makes this one unique without hardcoding it in the repo.
resource "aws_s3_bucket" "state" {
  bucket = "notes-rag-tfstate-${data.aws_caller_identity.current.account_id}"
}

# Versioning keeps every previous state, so a broken apply can be rolled back.
resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "state_bucket" {
  value = aws_s3_bucket.state.bucket
}
