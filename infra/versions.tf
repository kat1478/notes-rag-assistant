terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
  # Partial config: bucket is passed at init (scripts/deploy.sh) so the account ID stays out of the repo.
  backend "s3" {
    key          = "notes-rag/app.tfstate"
    region       = "eu-north-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "notes-rag", Env = "demo", ManagedBy = "terraform" }
  }
}
