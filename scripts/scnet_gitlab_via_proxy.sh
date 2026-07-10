#!/bin/bash
# SCNet 经集群 HTTP 代理访问 GitLab（直连 111.6.188.181:443 超时时使用）
#
# 根因（常见）:
#   - NO_PROXY 含 gitlab.eduxiji.net → curl/git 不走代理 → 连接超时
#   - 容器 DNS 不解析校外域名 → 需 /etc/hosts
#
# 用法:
#   bash scripts/scnet_gitlab_via_proxy.sh test
#   export GITLAB_TOKEN=glpat-xxx
#   bash scripts/scnet_gitlab_via_proxy.sh download-zip lutinayi_branch
#   bash scripts/scnet_gitlab_via_proxy.sh clone lutinayi_branch
set -euo pipefail

GITLAB_HOST="${GITLAB_HOST:-gitlab.eduxiji.net}"
GITLAB_IP="${GITLAB_IP:-111.6.188.181}"
PROXY="${SCNET_HTTPS_PROXY:-${https_proxy:-${HTTPS_PROXY:-http://preset:6e298f07@10.16.1.51:3128}}}"
REPO_PATH="fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu"
SCNET_HOME="${SCNET_HOME:-/public/home/xdzs2026_c415}"
DEST="${DEST:-$SCNET_HOME/2025pra-fdu-fudiankuangxiangqu}"

log() { echo "[via_proxy] $*"; }

_ensure_hosts() {
    if ! getent hosts "$GITLAB_HOST" >/dev/null 2>&1; then
        if [[ "$(id -u)" -eq 0 ]]; then
            echo "$GITLAB_IP $GITLAB_HOST" >> /etc/hosts
            log "added /etc/hosts"
        else
            echo "$GITLAB_IP $GITLAB_HOST" | sudo tee -a /etc/hosts >/dev/null
            log "added /etc/hosts (sudo)"
        fi
    fi
}

_use_proxy() {
    export http_proxy="$PROXY"
    export https_proxy="$PROXY"
    export HTTP_PROXY="$PROXY"
    export HTTPS_PROXY="$PROXY"
    # 关键：不要对 GitLab 使用 NO_PROXY，否则直连外网会超时
    export no_proxy="localhost,127.0.0.1,.zzai2.scnet.cn,zzai2.scnet.cn"
    export NO_PROXY="$no_proxy"
    unset ftp_proxy FTP_PROXY all_proxy ALL_PROXY 2>/dev/null || true
    log "proxy=$PROXY"
    log "no_proxy=$no_proxy"
}

_cmd_test() {
    _ensure_hosts
    _use_proxy
    log "curl homepage"
    curl -sSI -m 45 "https://${GITLAB_HOST}/" | head -12
    log "curl git info/refs (anonymous, expect 403 if private)"
    curl -sSI -m 45 \
        "https://${GITLAB_HOST}/${REPO_PATH}.git/info/refs?service=git-upload-pack" | head -12
}

_cmd_download_zip() {
    local branch="${1:-lutinayi_branch}"
    local token="${GITLAB_TOKEN:-${GL_TOKEN:-}}"
    local out="${2:-$SCNET_HOME/${REPO_PATH##*/}-${branch}.zip}"
    local url="https://${GITLAB_HOST}/${REPO_PATH}/-/archive/${branch}/${REPO_PATH##*/}-${branch}.zip"

    _ensure_hosts
    _use_proxy

    if [[ -z "$token" ]]; then
        log "WARN: 无 GITLAB_TOKEN，私有仓库会失败"
    fi

    log "download $url -> $out"
    if [[ -n "$token" ]]; then
        curl -fL -m 300 --retry 2 -H "PRIVATE-TOKEN: $token" -o "$out" "$url"
    else
        curl -fL -m 300 --retry 2 -o "$out" "$url"
    fi
    ls -lh "$out"
    unzip -l "$out" 2>/dev/null | head -6
}

_cmd_clone() {
    local branch="${1:-lutinayi_branch}"
    local token="${GITLAB_TOKEN:-${GL_TOKEN:-}}"
    if [[ -z "$token" ]]; then
        echo "ERROR: 私有仓库需要 GITLAB_TOKEN" >&2
        exit 1
    fi
    _ensure_hosts
    _use_proxy
    local url="https://oauth2:${token}@${GITLAB_HOST}/${REPO_PATH}.git"
    rm -rf "$DEST"
    git -c http.proxy="$PROXY" -c https.proxy="$PROXY" \
        clone --depth 1 -b "$branch" "$url" "$DEST"
    ls -la "$DEST/setup.py" "$DEST/vllm/__init__.py" 2>/dev/null
    log "clone OK -> $DEST"
}

main() {
    local action="${1:-test}"
    shift || true
    case "$action" in
        test) _cmd_test ;;
        download-zip|zip) _cmd_download_zip "$@" ;;
        clone) _cmd_clone "$@" ;;
        *)
            echo "Usage: scnet_gitlab_via_proxy.sh {test|download-zip [branch] [out]|clone [branch]}" >&2
            exit 1
            ;;
    esac
}

main "$@"
