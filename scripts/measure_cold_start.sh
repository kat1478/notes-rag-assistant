#!/usr/bin/env bash
# Measure Lambda cold starts: each round changes the function description, which makes Lambda discard warm
# instances, then times the first /health and the first /search (which loads the embedding model).
# Free: no /ask, no LLM call. The next deploy.sh resets the description.
# Usage: scripts/measure_cold_start.sh [rounds]
set -euo pipefail

export AWS_PROFILE="${AWS_PROFILE:-notes-rag}"
ROUNDS="${1:-5}"
URL=$(aws lambda get-function-url-config --function-name notes-rag --query FunctionUrl --output text)
URL="${URL%/}"

echo "round,cold_health_s,first_search_s,warm_health_s"
for n in $(seq 1 "$ROUNDS"); do
    aws lambda update-function-configuration --function-name notes-rag \
        --description "cold-start probe $n" --query LastUpdateStatus --output text >/dev/null
    aws lambda wait function-updated-v2 --function-name notes-rag
    cold=$(curl -s -o /dev/null -w "%{time_total}" --max-time 120 "$URL/health")
    search=$(curl -s -o /dev/null -w "%{time_total}" --max-time 60 -X POST -H 'Content-Type: application/json' \
        -d '{"query":"Co to jest algorytm Dijkstry?","top_k":3}' "$URL/search")
    warm=$(curl -s -o /dev/null -w "%{time_total}" "$URL/health")
    echo "$n,$cold,$search,$warm"
done
aws lambda update-function-configuration --function-name notes-rag --description "" \
    --query LastUpdateStatus --output text >/dev/null
