#!/bin/bash
# 修复 SCNet 容器内 GitLab DNS（常见根因：无法解析 gitlab.eduxiji.net）
set -euo pipefail

GITLAB_HOST="${GITLAB_HOST:-gitlab.eduxiji.net}"
GITLAB_IP="${GITLAB_IP:-111.6.188.181}"

if grep -q "$GITLAB_HOST" /etc/hosts 2>/dev/null; then
  echo "[fix_hosts] 已存在: $(grep "$GITLAB_HOST" /etc/hosts)"
  exit 0
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "需要 root。请执行:"
  echo "  echo '$GITLAB_IP $GITLAB_HOST' | sudo tee -a /etc/hosts"
  exit 1
fi

echo "$GITLAB_IP $GITLAB_HOST" >> /etc/hosts
echo "[fix_hosts] OK: $GITLAB_IP $GITLAB_HOST"
getent hosts "$GITLAB_HOST" || true
