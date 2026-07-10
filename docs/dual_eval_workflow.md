# 双通道评测：官方平台 vs SCNet

> **问题**：官方提交处有权限但**无详细报错**；SCNet 能调试但 **GitLab clone 403**、不能代替平台打分。  
> **解法**：平台负责提交得分；SCNet 用 zip 导入 + `platform_build.sh` **镜像编译并保存完整 log**。

---

## 两套环境对照

| | 官方竞赛平台 | SCNet 容器 |
|--|-------------|------------|
| **用途** | 正式提交、得分 | 本地复现编译/启动、拿详细 log |
| **你的权限** | 富贵花开 ✅ 可提交 | xdzs2026_c415 ✅ 可 SSH |
| **GitLab** | 平台自动拉 `main` | `git clone` 常 **403** ❌ |
| **报错** | 仅「missing setup.py」「vLLM build failed」 | 跑脚本可得**完整编译 log** |
| **打分** | ✅ 唯一有效 | ❌ 仅自测 |

**不要混用**：SCNet 分数不上榜；平台 log 不够时**必须**在 SCNet 镜像编译。

---

## 分工（谁做什么）

| 步骤 | 谁 | 在哪 |
|------|-----|------|
| 改代码 + push GitLab `main` | 队友 / 本地 | Windows + GitLab |
| **平台提交评测** | **富贵花开** | pra.xtnl.org.cn / course.educg.net |
| zip 下载最新 `main` | 你 | GitLab 网页 → Download ZIP |
| zip 传到 SCNet | 你 | 网页上传 / `scp` / 共享盘 |
| **镜像平台编译 + 保存 log** | 你 | SCNet：`platform_build.sh` |
| 修编译问题后再 push | 队友 | GitLab → 再平台提交 |

---

## A. 官方平台提交（得分）

1. 登录竞赛平台（**富贵花开**）
2. 仓库：`https://gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu.git`
3. 分支：**`main`**
4. 确认 commit ≥ **`08e608b`**（含 `requirements/`）
5. 提交评测

若只显示 `vLLM build failed` → 走 **B 节** 拿 log，修完再提交。

---

## B. SCNet 无 Git 权限：zip 导入 + 镜像编译

### B1. Windows 准备 zip（任选其一）

**方式 1 — GitLab 网页（推荐）**

1. 打开仓库 → 分支 **`main`**
2. **Download ZIP**
3. 确认 zip 解压后有 `setup.py`、`vllm/`、`requirements/`（旧 zip 仅 48 个文件无效）

**方式 2 — 本地打包（队友已 push 后）**

```powershell
cd C:\Users\Lucifer\Desktop\compute\baseline
git fetch origin
git archive --format=zip --output "$env:USERPROFILE\Downloads\main-latest.zip" origin/main
```

### B2. 把 zip 弄进 SCNet

任选：

- SCNet 网页 **文件上传** 到家目录
- 或本机 `scp` 到 `/public/home/xdzs2026_c415/`

### B3. SCNet 终端（复制粘贴）

**若还没有仓库目录**（首次），先手动解压 zip 再进目录：

```bash
cd /public/home/xdzs2026_c415
unzip -qo ./main-latest.zip
rm -rf ./2025pra-fdu-fudiankuangxiangqu
mv ./2025pra-fdu-fudiankuangxiangqu-main ./2025pra-fdu-fudiankuangxiangqu
cd ./2025pra-fdu-fudiankuangxiangqu
ls setup.py vllm/__init__.py requirements/rocm.txt
bash scripts/platform_build.sh
```

**若已有旧仓库**（含 `scripts/`），可用导入脚本覆盖：

```bash
cd /public/home/xdzs2026_c415/2025pra-fdu-fudiankuangxiangqu
bash scripts/scnet_import_repo.sh /public/home/xdzs2026_c415/main-latest.zip
bash scripts/platform_build.sh
```

成功后终端会打印：

```
LOG=/public/home/xdzs2026_c415/2025pra-fdu-fudiankuangxiangqu/results/platform_build_YYYYMMDD_HHMMSS.log
```

**把该 log 全文复制给队友** — 等价于官方平台拿不到的编译详情。

### B4. 编译通过后（可选）测 launch

```bash
cd /public/home/xdzs2026_c415/2025pra-fdu-fudiankuangxiangqu/dist
pip install vllm-*.whl --no-deps

cp -r /public/home/xdzs2026_c415/Qwen3.5-27B /root/Qwen3.5-27B
bash /public/home/xdzs2026_c415/2025pra-fdu-fudiankuangxiangqu/scripts/scnet_start_optimized.sh
```

---

## C. 一键恢复（已有仓库、仅重编）

```bash
cd /public/home/xdzs2026_c415/2025pra-fdu-fudiankuangxiangqu
git pull origin main   # 若 403 则改用 B 节 zip 导入
bash scripts/platform_build.sh
```

---

## D. 常见错误对照

| 官方平台提示 | SCNet 自查 | 处理 |
|-------------|-----------|------|
| missing setup.py | `ls setup.py vllm/__init__.py` | 换新版 zip / push `main` |
| vLLM build failed | `bash scripts/platform_build.sh` 看 log | 按 log 修依赖/ROCm/源码 |
| git clone 403 | 用 zip + `scnet_import_repo.sh` | 不要硬 clone |

---

## E. GitLab 403 的长期解法（可选）

由仓库 Owner 在 GitLab 创建 **Deploy Token**（`read_repository`），SCNet 可用：

```bash
curl -fL -o main.zip \
  "https://<token-name>:<token>@gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu/-/archive/main/2025pra-fdu-fudiankuangxiangqu-main.zip"
bash scripts/scnet_import_repo.sh ./main.zip
```

Token 勿提交进 git。

---

## 相关脚本

| 脚本 | 作用 |
|------|------|
| `scripts/platform_build.sh` | 镜像 `/coursegrader/submit` 编译，写 log |
| `scripts/scnet_import_repo.sh` | zip 导入，绕过 git 403 |
| `scripts/scnet_resume.sh` | 容器重启后恢复 wheel + 服务 |
| `scripts/prepare_submit.sh` | 重新 vendor vllm 到根目录 |
