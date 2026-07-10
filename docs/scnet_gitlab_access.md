# SCNet ↔ GitLab 权限问题排查手册

> 在 SCNet 容器 SSH 里执行诊断脚本，根据输出对号入座。  
> 脚本：`bash scripts/scnet_gitlab_diagnose.sh 2>&1 | tee ~/gitlab_diagnose.log`

---

## 所有可能原因（按概率）

| # | 现象 | 根因 | 验证命令 | 解法 |
|---|------|------|----------|------|
| **1** | `Could not resolve host: gitlab.eduxiji.net` | **容器 DNS 不解析校外域名** | `getent hosts gitlab.eduxiji.net` 失败 | `/etc/hosts` 或 `--resolve`（见下） |
| **2** | URL 变成 `gitlab.eduxiji.ne` | **复制粘贴截断** | 检查命令里域名是否完整 `.net` | 用变量 `U="https://..."` 再 `curl "$U"` |
| **3** | `HTTP 302` → `sign_in` / `401` / `403` | **仓库私有，匿名无 read** | 诊断 §3 ZIP 返回登录页 | **Deploy Token** 或 **ZIP 网页下载上传** |
| **4** | `403` + `Connection established` + proxy | **http_proxy 劫持 Git 鉴权** | `env \| grep -i proxy` | `unset` 全部 proxy 再 git/curl |
| **5** | `403` 无 proxy | **Token 错 / 过期 / 无项目权限** | 用 token 仍 403 | Owner 加成员或新建 Deploy Token |
| **6** | `Connection timed out` | **容器出网封禁教育网 GitLab** | `nc 111.6.188.181 443` 失败 | **只能 ZIP 上传**，放弃 curl/git |
| **7** | `SSL certificate problem` | 容器 CA 不全 | curl 报 cert | `GIT_SSL_NO_VERIFY=true`（仅调试，不推荐长期） |
| **8** | git 403 但 curl IP 通 | **git 走 proxy、cURL 不走** | 对比 §4/§5 | 统一 `unset proxy` + `git -c http.proxy=` |
| **9** | 竞赛平台能拉、SCNet 不能 | **两环境网络策略不同** | 平台有内网镜像 | SCNet 用 ZIP，不依赖 GitLab |
| **10** | `git clone` 成功无 `setup.py` | **分支错 / 旧 commit** | `git log -1` | 换 `main` 或 `lutinayi_branch` 最新 |

**你当前日志**（`Could not resolve host: gitlab.eduxiji.ne`）→ 优先查 **#1 DNS** + **#2 截断**。

---

## 一键诊断（你在 SCNet 执行）

```bash
cd /public/home/xdzs2026_c415

# 若还没有脚本，先把 Windows zip 解压后的仓库弄进来，或从队友处 scp diagnose 脚本
# 有仓库后：
cd 2025pra-fdu-fudiankuangxiangqu   # 或任意含 scripts/ 的目录
bash scripts/scnet_gitlab_diagnose.sh 2>&1 | tee ~/gitlab_diagnose.log
```

没有仓库时，可只跑精简版：

```bash
cd /public/home/xdzs2026_c415
GITLAB_HOST=gitlab.eduxiji.net
GITLAB_IP=111.6.188.181

echo "=== DNS ==="
getent hosts $GITLAB_HOST || echo FAIL_DNS
grep $GITLAB_HOST /etc/hosts || echo NO_HOSTS

echo "=== curl 域名 ==="
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
curl -sSI -m 15 "https://${GITLAB_HOST}/" | head -5

echo "=== curl --resolve ==="
curl -sSI -m 15 --resolve "${GITLAB_HOST}:443:${GITLAB_IP}" "https://${GITLAB_HOST}/" | head -5

echo "=== TCP ==="
nc -zv -w 5 $GITLAB_IP 443 2>&1 || echo NC_FAIL
```

**把 `~/gitlab_diagnose.log` 全文贴回。**

---

## 修复步骤（按诊断结果）

### 修复 A：DNS（#1）— 最常见

```bash
# 容器里多为 root；若是 xdzs 用户加 sudo
echo '111.6.188.181 gitlab.eduxiji.net' | sudo tee -a /etc/hosts
getent hosts gitlab.eduxiji.net

# 再试下载（域名）
curl -fL -o /tmp/test.zip \
  "https://gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu/-/archive/lutinayi_branch/2025pra-fdu-fudiankuangxiangqu-lutinayi_branch.zip"
ls -lh /tmp/test.zip
```

或不用改 hosts，用 **--resolve**：

```bash
curl -fL --resolve "gitlab.eduxiji.net:443:111.6.188.181" -o lutinayi.zip \
  "https://gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu/-/archive/lutinayi_branch/2025pra-fdu-fudiankuangxiangqu-lutinayi_branch.zip"
```

### 修复 B：清 proxy（#4）

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
export NO_PROXY=127.0.0.1,localhost,gitlab.eduxiji.net,111.6.188.181
git -c http.proxy= -c https.proxy= clone --depth 1 -b lutinayi_branch \
  https://gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu.git
```

### 修复 C：私有仓库 Token（#3/#5）

GitLab 网页（你有 Owner 权限）→ **Settings → Access Tokens** → `read_repository`

```bash
export GITLAB_TOKEN='你的token'
bash scripts/scnet_gitlab_clone.sh lutinayi_branch
```

或 Deploy Token（Settings → Repository → Deploy tokens）。

### 修复 D：完全无法出网（#6）— 你很可能最终落这里

```bash
# Windows 已下载:
# C:\Users\Lucifer\Downloads\2025pra-fdu-fudiankuangxiangqu-lutinayi_branch.zip
# → SCNet 网页上传或 rz 到 /public/home/xdzs2026_c415/
unzip -qo ./2025pra-fdu-fudiankuangxiangqu-lutinayi_branch.zip
mv ./2025pra-fdu-fudiankuangxiangqu-lutinayi_branch ./2025pra-fdu-fudiankuangxiangqu
```

**不依赖 GitLab 网络权限。**

---

## 决策树

```
getent hosts gitlab.eduxiji.net 失败?
  ├─ 是 → fix_hosts / --resolve → 再 curl
  │        ├─ 200/302 到 zip → 若 403/登录 → Token 或 ZIP 上传
  │        └─ timeout → 出网封禁 → 只能 ZIP 上传
  └─ 否 → unset proxy → git clone
           ├─ 403 → Token
           └─ 200 → 成功
```

---

## 与「官方平台」的关系

| 环境 | GitLab | 说明 |
|------|--------|------|
| 竞赛平台 | ✅ 内网拉仓库 | 所以有提交权限就能编，但 log 简 |
| SCNet 容器 | ❌ 常 DNS/403/封网 | **不等于**平台权限；用 ZIP 镜像编译 |

---

## 相关脚本

| 脚本 | 作用 |
|------|------|
| `scnet_gitlab_diagnose.sh` | 全量探测，输出 PASS/FAIL |
| `scnet_gitlab_fix_hosts.sh` | 写入 `111.6.188.181 gitlab.eduxiji.net` |
| `scnet_gitlab_clone.sh` | `GITLAB_TOKEN` 克隆 |
| `scnet_import_repo.sh` | 本地 zip 导入 |
| `platform_build.sh` | 镜像平台编译 log |
