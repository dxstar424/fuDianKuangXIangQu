#!/bin/bash
# SCNet ↔ GitLab 权限/连通性一键诊断（在 SCNet 容器内执行）
# 用法: bash scnet_gitlab_diagnose.sh 2>&1 | tee ~/gitlab_diagnose.log
set -u
set +e

REPO="${REPO:-https://gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu.git}"
BRANCH="${BRANCH:-lutinayi_branch}"
ZIP_URL="${ZIP_URL:-https://gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu/-/archive/${BRANCH}/2025pra-fdu-fudiankuangxiangqu-${BRANCH}.zip}"
GITLAB_HOST="${GITLAB_HOST:-gitlab.eduxiji.net}"
GITLAB_IP="${GITLAB_IP:-111.6.188.181}"
LOG="${LOG:-$HOME/gitlab_diagnose.log}"

section() { echo ""; echo "========== $* =========="; }
pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; }
warn() { echo "[WARN] $*"; }
info() { echo "[INFO] $*"; }

section "0. 环境"
info "user=$(whoami) host=$(hostname) pwd=$PWD"
info "date=$(date -Is 2>/dev/null || date)"
info "home=$HOME"
env | grep -iE 'proxy|PROXY|no_proxy|NO_PROXY' || info "无 proxy 环境变量"

section "1. DNS"
if command -v getent >/dev/null 2>&1; then
  getent hosts "$GITLAB_HOST" && pass "getent $GITLAB_HOST" || fail "getent $GITLAB_HOST"
else
  warn "无 getent"
fi
if command -v nslookup >/dev/null 2>&1; then
  nslookup "$GITLAB_HOST" 2>&1 | head -8
elif command -v host >/dev/null 2>&1; then
  host "$GITLAB_HOST" 2>&1
else
  warn "无 nslookup/host"
fi
if grep -q "$GITLAB_HOST" /etc/hosts 2>/dev/null; then
  pass "/etc/hosts 有 $GITLAB_HOST 条目"
  grep "$GITLAB_HOST" /etc/hosts
else
  warn "/etc/hosts 无 $GITLAB_HOST（可手动添加: echo '$GITLAB_IP $GITLAB_HOST' >> /etc/hosts）"
fi
cat /etc/resolv.conf 2>/dev/null | head -6 || warn "无 resolv.conf"

section "2. TCP 连通 (443)"
if command -v curl >/dev/null 2>&1; then
  curl -sS -m 10 -o /dev/null -w "curl_host code=%{http_code} time=%{time_total}s\n" "https://${GITLAB_HOST}/" || fail "curl 域名 HTTPS"
  curl -sS -m 10 -o /dev/null -w "curl_ip   code=%{http_code} time=%{time_total}s\n" --resolve "${GITLAB_HOST}:443:${GITLAB_IP}" "https://${GITLAB_HOST}/" || fail "curl --resolve IP HTTPS"
else
  fail "无 curl"
fi
if command -v nc >/dev/null 2>&1; then
  nc -zv -w 5 "$GITLAB_IP" 443 2>&1 && pass "nc $GITLAB_IP:443" || fail "nc $GITLAB_IP:443"
fi

section "3. HTTP 探测（无鉴权）"
for u in \
  "https://${GITLAB_HOST}/" \
  "https://${GITLAB_HOST}/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu" \
  "$ZIP_URL"
do
  info "HEAD $u"
  curl -sSI -m 15 -L "$u" 2>&1 | head -12
  echo "---"
done

section "4. ZIP 下载（匿名）"
TMPZIP="/tmp/gitlab_test_$$.zip"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="127.0.0.1,localhost,${GITLAB_HOST},${GITLAB_IP}"

info "尝试1: 域名"
curl -fL -m 60 --retry 1 -o "$TMPZIP" "$ZIP_URL" 2>&1
RC1=$?
if [[ $RC1 -eq 0 && -s "$TMPZIP" ]]; then
  pass "ZIP 域名下载 size=$(stat -c%s "$TMPZIP" 2>/dev/null || wc -c <"$TMPZIP")"
  unzip -l "$TMPZIP" 2>/dev/null | head -5
else
  fail "ZIP 域名下载 rc=$RC1"
  rm -f "$TMPZIP"
  info "尝试2: --resolve 固定 IP"
  curl -fL -m 60 --retry 1 --resolve "${GITLAB_HOST}:443:${GITLAB_IP}" -o "$TMPZIP" "$ZIP_URL" 2>&1
  RC2=$?
  if [[ $RC2 -eq 0 && -s "$TMPZIP" ]]; then
    pass "ZIP --resolve 下载 size=$(stat -c%s "$TMPZIP" 2>/dev/null || wc -c <"$TMPZIP")"
  else
    fail "ZIP --resolve 下载 rc=$RC2"
  fi
fi
rm -f "$TMPZIP"

section "5. Git clone（匿名 HTTPS）"
WORKDIR="/tmp/gitlab_clone_test_$$"
rm -rf "$WORKDIR"
info "git clone --depth 1 -b $BRANCH $REPO"
GIT_CURL_VERBOSE=1 GIT_TRACE_CURL=1 \
  git -c http.proxy= -c https.proxy= clone --depth 1 -b "$BRANCH" "$REPO" "$WORKDIR" 2>&1 | tail -30
RCG=$?
if [[ $RCG -eq 0 && -f "$WORKDIR/setup.py" ]]; then
  pass "git clone OK, setup.py exists"
elif [[ $RCG -eq 0 ]]; then
  warn "git clone OK 但无 setup.py（分支/内容问题）"
else
  fail "git clone rc=$RCG"
fi
rm -rf "$WORKDIR"

section "6. Git clone（--resolve / 改 hosts 后再试）"
if ! grep -q "$GITLAB_HOST" /etc/hosts 2>/dev/null; then
  info "临时写入 /etc/hosts（需 root）"
  if [[ "$(id -u)" -eq 0 ]]; then
    echo "$GITLAB_IP $GITLAB_HOST" >> /etc/hosts
    pass "已写入 /etc/hosts"
  else
    warn "非 root，跳过 hosts 写入；可 sudo 执行: echo '$GITLAB_IP $GITLAB_HOST' | sudo tee -a /etc/hosts"
  fi
fi
rm -rf "$WORKDIR"
git -c http.proxy= -c https.proxy= clone --depth 1 -b "$BRANCH" "$REPO" "$WORKDIR" 2>&1 | tail -20
RCG2=$?
[[ $RCG2 -eq 0 ]] && pass "hosts 修复后 git clone" || fail "hosts 修复后 git clone rc=$RCG2"
rm -rf "$WORKDIR"

section "7. 鉴权方式提示（需人工 token）"
cat <<'EOF'
若上文 ZIP/git 返回 302→sign_in 或 401/403：
  原因: 仓库私有，匿名无 read 权限（不是 SCNet 网络问题）
  解法:
    A) GitLab → Settings → Access Tokens → read_repository
       git clone https://oauth2:<TOKEN>@gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu.git
    B) Deploy Token（Owner 创建）→ 同上
    C) 不用 git：Windows 下载 ZIP → SCNet 网页/rz 上传 → scnet_import_repo.sh
若 DNS 失败但 --resolve / hosts 成功：
  原因: 容器 DNS 不解析校外域名
  解法: 永久 echo '111.6.188.181 gitlab.eduxiji.net' >> /etc/hosts（每次新容器重做）
若全部网络 FAIL：
  原因: 容器出网策略禁止访问教育网 GitLab
  解法: 只能 ZIP 上传，无法 curl/git
EOF

section "8. 结论摘要"
echo "完整日志: $LOG"
echo "请把本脚本输出全文发给队友排查。"
