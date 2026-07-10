#!/bin/bash
# SCNet 无 GitLab clone 权限（403）时，用 zip 导入仓库。
#
# 用法:
#   bash scripts/scnet_import_repo.sh /path/to/2025pra-fdu-fudiankuangxiangqu-main.zip
#   bash scripts/scnet_import_repo.sh --url 'https://gitlab.../-/archive/main/xxx-main.zip'
#
# 环境变量:
#   PROJ          目标目录（默认 $SCNET_HOME/2025pra-fdu-fudiankuangxiangqu）
#   SCNET_HOME    默认 /public/home/xdzs2026_c415
set -euo pipefail

SCNET_HOME="${SCNET_HOME:-/public/home/xdzs2026_c415}"
PROJ="${PROJ:-$SCNET_HOME/2025pra-fdu-fudiankuangxiangqu}"
STAGING="${STAGING:-/tmp/fdu_repo_import_$$}"

_cleanup() { rm -rf "$STAGING"; }
trap _cleanup EXIT

log() { echo "[scnet_import] $*"; }

_fetch_zip() {
    local src="$1"
    local dest="$STAGING/repo.zip"
    mkdir -p "$STAGING"
    if [[ "$src" == --url=* ]]; then
        local url="${src#--url=}"
        log "curl download $url"
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
        curl -fL --retry 3 -o "$dest" "$url"
    elif [[ "$src" == http://* || "$src" == https://* ]]; then
        log "curl download $src"
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
        curl -fL --retry 3 -o "$dest" "$src"
    elif [[ -f "$src" ]]; then
        cp -f "$src" "$dest"
    else
        echo "ERROR: zip 不存在: $src" >&2
        exit 1
    fi
    echo "$dest"
}

_unpack() {
    local zip="$1"
    local out="$STAGING/unpack"
    mkdir -p "$out"
    unzip -qo "$zip" -d "$out"
    local top
    top="$(find "$out" -mindepth 1 -maxdepth 1 -type d | head -1)"
    if [[ -z "$top" || ! -f "$top/setup.py" ]]; then
        echo "ERROR: zip 内无 setup.py，可能是旧版包（缺 vllm 源码）" >&2
        echo "  需要 commit >= 08e608b，含 setup.py + vllm/ + requirements/" >&2
        exit 2
    fi
    echo "$top"
}

_install() {
    local src="$1"
    log "install $src -> $PROJ"
    mkdir -p "$(dirname "$PROJ")"
    rm -rf "$PROJ"
    cp -a "$src" "$PROJ"
    chmod +x "$PROJ"/launch.sh "$PROJ"/scripts/*.sh 2>/dev/null || true
    log "OK: $PROJ"
    log "  setup.py: $(test -f "$PROJ/setup.py" && echo yes || echo NO)"
    log "  vllm/:    $(test -d "$PROJ/vllm" && echo yes || echo NO)"
    log "  requirements/: $(test -d "$PROJ/requirements" && echo yes || echo NO)"
    echo ""
    echo "下一步（拿官方平台同款编译 log）:"
    echo "  cd $PROJ && bash scripts/platform_build.sh"
}

main() {
    if [[ $# -lt 1 ]]; then
        echo "Usage: scnet_import_repo.sh <zip-file|--url=URL>" >&2
        exit 1
    fi
    local zip_path
    zip_path="$(_fetch_zip "$1")"
    local root
    root="$(_unpack "$zip_path")"
    _install "$root"
}

main "$@"
