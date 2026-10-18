#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_KIND="${1:-lightnav}"
if [[ $# -gt 0 ]]; then shift; fi
export CUDA_VISIBLE_DEVICES="${MODEL_GPU:-1}"
case "$MODEL_KIND" in
  lightnav)
    export PYTHONPATH="$PWD/src"
    export PORT="${PORT:-8050}"
    exec "$HOME/.conda/envs/lightnav/bin/python" -m lightnav.serving.ws_server \
      --task vln --model_path "$PWD/checkpoints/LightNav-0" --backend vllm_local \
      --gpu_memory_utilization 0.75 "$@"
    ;;
  omnivla)
    export PYTHONPATH="$PWD/habitat_demo"
    exec "$HOME/.conda/envs/omnivla/bin/python" -m habitat_web.omnivla_service \
      --repo "$PWD/../OmniVLA" --checkpoint "$PWD/../OmniVLA/omnivla-original" \
      --host 127.0.0.1 --port 8891 "$@"
    ;;
  *) echo "Usage: $0 {lightnav|omnivla} [model service arguments]" >&2; exit 2 ;;
esac
