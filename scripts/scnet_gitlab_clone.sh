#!/bin/bash
# 使用 GitLab Token 克隆（私有仓库 403 时）
# 用法:
#   export GITLAB_TOKEN='glpat-xxxx'   # 或 Deploy Token
#   bash scripts/scnet_gitlab_clone.sh [branch]
set -euo pipefail

BRANCH="${1:-lutinayi_branch}"
TOKEN="${GITLAB_TOKEN:-${GL_TOKEN:-}}"
USER="${GITLAB_USER:-oauth2}"
REPO_PATH="fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu"
HOST="${GITLAB_HOST:-gitlab.eduxiji.net}"
SCNET_HOME="${SCNET_HOME:-/public/home/xdzs2026_c415}"
DEST="${DEST:-$SCNET_HOME/2025pra-fdu-fudiankuangxiangqu}"

if [[ -z "$TOKEN" ]]; then
  echo "ERROR: 设置 GITLAB_TOKEN（GitLab → Settings → Access Tokens → read_repository）" >&2
  exit 1
fi

bash "$(dirname "$0")/scnet_gitlab_fix_hosts.sh" 2>/dev/null || true
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,$HOST}"

URL="https://${USER}:${TOKEN}@${HOST}/${REPO_PATH}.git"
echo "[clone] branch=$BRANCH dest=$DEST"
rm -rf "$DEST"
git -c http.proxy= -c https.proxy= clone --depth 1 -b "$BRANCH" "$URL" "$DEST"
ls -la "$DEST/setup.py" "$DEST/vllm/__init__.py" "$DEST/requirements/rocm.txt"
echo "[clone] OK"
