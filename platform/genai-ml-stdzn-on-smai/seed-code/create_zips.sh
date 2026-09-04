#!/bin/bash
# Builds the seed-code zips that SageMaker projects check into the shared
# model-build and model-deploy repositories.
#
# One folder here per kind of model code. Every folder that holds a model_build
# and a model_deploy subfolder is picked up automatically, so adding a new kind
# of model means adding a folder and running this script again.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Each subfolder is copied into the model folder as-is, so the zip has to keep
# the wrapper folder. A workflow file at the root of a repository can only be
# pushed by a GitHub App that holds the workflows permission, and the seeding
# App does not, which is the other reason the wrapper matters.
build_zip() {
  local kind_dir=$1 subfolder=$2 zip_name=$3
  local staging
  staging=$(mktemp -d)

  cp -r "$kind_dir/$subfolder" "$staging/"

  # zip appends to an existing archive, so remove it first to get a clean build
  rm -f "$kind_dir/$zip_name"
  ( cd "$staging" && zip -rq "$kind_dir/$zip_name" "$subfolder" \
      -x "*__pycache__/*" -x "*.pyc" -x "*.DS_Store" )

  rm -rf "$staging"
  echo "  $zip_name  ($(unzip -Z1 "$kind_dir/$zip_name" | wc -l | tr -d ' ') entries)"
}

found=0
for kind_dir in "$SCRIPT_DIR"/*/; do
  kind_dir=${kind_dir%/}
  [ -d "$kind_dir/model_build" ] && [ -d "$kind_dir/model_deploy" ] || continue

  found=1
  echo "$(basename "$kind_dir"):"
  build_zip "$kind_dir" model_build model-build-repo.zip
  build_zip "$kind_dir" model_deploy model-deploy-repo.zip
done

if [ "$found" -eq 0 ]; then
  echo "No seed-code folders found. A folder needs both model_build and model_deploy." >&2
  exit 1
fi
