#!/bin/sh
set -e
# Chroma opens its SQLite file read-write and Lambda only allows writes in /tmp, so work on a copy of the baked
# index. A re-initialised sandbox may still have it from the previous start.
[ -d "$CHROMA_DIR" ] || cp -r /app/backend/data/chroma "$CHROMA_DIR"

# The key lives in SSM (SecureString), not in the image or the function config, so it never shows in the console
# or Terraform state. Without it the app still serves retrieval with the extractive fallback — log loudly.
if [ -n "$OPENROUTER_KEY_PARAMETER" ] && [ -z "$OPENROUTER_API_KEY" ]; then
    if OPENROUTER_API_KEY=$(python -c "import boto3, os; print(boto3.client('ssm').get_parameter(Name=os.environ['OPENROUTER_KEY_PARAMETER'], WithDecryption=True)['Parameter']['Value'])"); then
        export OPENROUTER_API_KEY
    else
        echo "ERROR: could not read $OPENROUTER_KEY_PARAMETER from SSM; /ask will use the extractive fallback" >&2
    fi
fi
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
