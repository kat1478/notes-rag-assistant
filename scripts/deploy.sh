#!/usr/bin/env bash
# Build the Lambda image, push it to ECR and apply Terraform.
# Needs: `aws login --profile notes-rag --remote`, Docker, Terraform >= 1.10, and the state bucket (infra/bootstrap).
# Terraform prints its plan and asks before changing anything.
set -euo pipefail

export AWS_PROFILE="${AWS_PROFILE:-notes-rag}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Terraform's built-in S3 backend doesn't support `aws login` sessions yet (the AWS provider does), so hand it
# short-lived keys from the CLI. Fresh on every call (they expire within minutes and the image push is longer),
# and scoped to a subshell so later `aws` commands keep using the refreshing login session.
tf() (
    eval "$(aws configure export-credentials --format env)"
    exec terraform -chdir="$ROOT/infra" "$@"
)

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
tf init -input=false -reconfigure -backend-config="bucket=notes-rag-tfstate-$ACCOUNT" >/dev/null

# 1. Repository first: the function can only be created once an image exists in it.
tf apply -target=aws_ecr_repository.app -target=aws_ecr_lifecycle_policy.app
REPO=$(tf output -raw repository_url)

# 2. Build and push. Tags are immutable, so each deploy gets commit + timestamp.
TAG="$(git -C "$ROOT" rev-parse --short HEAD)-$(date +%Y%m%d%H%M%S)"
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%%/*}"
# --provenance=false: Lambda rejects the multi-manifest index that BuildKit attestations produce.
docker build --platform linux/amd64 --provenance=false --target lambda -t "$REPO:$TAG" "$ROOT"
docker push "$REPO:$TAG"
DIGEST=$(aws ecr describe-images --repository-name notes-rag --image-ids imageTag="$TAG" \
    --query 'imageDetails[0].imageDigest' --output text)

# 3. Function, URL, permissions — pinned to the digest, so a re-pushed tag can never change what runs.
tf apply -var "image_digest=$DIGEST"
echo "Deployed $TAG → $(tf output -raw function_url)"
