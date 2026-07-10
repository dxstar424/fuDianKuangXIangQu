#!/bin/bash
# 经 SCNet 代理 + GitLab Token 拉取仓库（zip 或 git）
set -euo pipefail

SCNET_HOME="${SCNET_HOME:-/public/home/xdzs2026_c415}"
PROXY="${SCNET_HTTPS_PROXY:-http://preset:6e298f07@10.16.1.51:3128}"
HOST="${GITLAB_HOST:-gitlab.eduxiji.net}"
HOST_IP="${GITLAB_IP:-111.6.188.181}"
PROJECT_ENC="fudiankuangxiangqu%2F2025pra-fdu-fudiankuangxiangqu"
REPO_PATH="fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu"
BRANCH="${1:-lutinayi_branch}"
MODE="${2:-auto}"
TOKEN="${GITLAB_TOKEN:-${GL_TOKEN:-}}"
DEST="${DEST:-$SCNET_HOME/2025pra-fdu-fudiankuangxiangqu}"
ZIP_OUT="${ZIP_OUT:-$SCNET_HOME/lutinayi.zip}"

log() { echo "[fetch] $*"; }
die() { echo "[fetch] ERROR: $*" >&2; exit 1; }

[[ -n "$TOKEN" ]] || die "export GITLAB_TOKEN=glpat-xxx  (read_repository)"

_setup() {
    grep -q "$HOST" /etc/hosts 2>/dev/null || echo "$HOST_IP $HOST" | tee -a /etc/hosts >/dev/null
    export http_proxy="$PROXY" https_proxy="$PROXY" HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
    export no_proxy="localhost,127.0.0.1,.zzai2.scnet.cn,zzai2.scnet.cn"
    unset NO_PROXY
}

_check_zip() {
    local f="$1"
    [[ -s "$f" ]] || return 1
    local sz
    sz=$(stat -c%s "$f" 2>/dev/null || wc -c <"$f")
    (( sz > 500000 )) || return 1
    unzip -t "$f" >/dev/null 2>&1
}

_show_bad_file() {
    local f="$1"
    log "下载文件仅 $(stat -c%s "$f" 2>/dev/null || wc -c <"$f") bytes，不是有效 zip"
    log "前 200 字节:"
    head -c 200 "$f" | tr '\0' ' ' ; echo
    log "若含 sign_in / 401 / 403 / error → Token 无效或权限不足"
}

_try_api_zip() {
    local url="https://${HOST}/api/v4/projects/${PROJECT_ENC}/repository/archive.zip?sha=${BRANCH}"
    log "API zip: $url"
    curl -fL -m 300 --retry 2 -H "PRIVATE-TOKEN: $TOKEN" -o "$ZIP_OUT" "$url"
}

_try_web_zip() {
    local url="https://${HOST}/${REPO_PATH}/-/archive/${BRANCH}/${REPO_PATH##*/}-${BRANCH}.zip"
    log "Web zip: $url"
    curl -fL -m 300 --retry 2 -H "PRIVATE-TOKEN: $TOKEN" -o "$ZIP_OUT" "$url"
}

_try_git_clone() {
    local url="https://oauth2:${TOKEN}@${HOST}/${REPO_PATH}.git"
    log "git clone -b $BRANCH"
    rm -rf "$DEST"
    git -c http.proxy="$PROXY" -c https.proxy="$PROXY" clone --depth 1 -b "$BRANCH" "$url" "$DEST"
    ls -la "$DEST/setup.py" "$DEST/vllm/__init__.py" "$DEST/requirements/rocm.txt"
    log "git clone OK -> $DEST"
}

_install_zip() {
    rm -rf "$DEST"
    local top
    top=$(unzip -Z1 "$ZIP_OUT" | head -1 | cut -d/ -f1)
    unzip -qo "$ZIP_OUT" -d "$SCNET_HOME"
    mv "$SCNET_HOME/$top" "$DEST"
    ls -la "$DEST/setup.py" "$DEST/vllm/__init__.py" "$DEST/requirements/rocm.txt"
    log "zip install OK -> $DEST"
}

_setup

case "$MODE" in
  git)
    _try_git_clone
    ;;
  zip-api)
    _try_api_zip
    _check_zip "$ZIP_OUT" || { _show_bad_file "$ZIP_OUT"; exit 2; }
    _install_zip
    ;;
  zip-web)
    _try_web_zip
    _check_zip "$ZIP_OUT" || { _show_bad_file "$ZIP_OUT"; exit 2; }
    _install_zip
    ;;
  auto|*)
    log "=== try API zip ==="
    if _try_api_zip && _check_zip "$ZIP_OUT"; then
        _install_zip; exit 0
    fi
    _show_bad_file "$ZIP_OUT" 2>/dev/null || true
    log "=== try git clone ==="
    if _try_git_clone; then exit 0; fi
    log "=== try web zip ==="
    _try_web_zip
    _check_zip "$ZIP_OUT" || { _show_bad_file "$ZIP_OUT"; exit 2; }
    _install_zip
    ;;
esac
