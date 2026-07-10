# SCNet 跑结果手册（官方 PDF + lutinayi_branch）

家目录：`/public/home/xdzs2026_c415`（容器重启后数据保留）

---

## 两套流程（别混）

| 目的 | 用什么 | 端口 |
|------|--------|------|
| **官方 baseline 自测** | PDF `start_vllm.sh`（stock vLLM） | 8001 |
| **lutinayi 优化版自测** | 仓库 `scnet_start_optimized.sh` → `launch.sh` | 8001 |
| **正式评测得分** | 竞赛平台提交 `lutinayi_branch`，跑 `launch.sh` | 平台定 |

**禁止再建议或执行 baseline `start_vllm.sh` 自测**（用户只测 `lutinayi_branch` 优化版）。

**baseline 不需要 GitLab 代码。优化版才需要 clone 仓库。**

---

## 一、环境准备（PDF 第 7–9 步，家目录执行）

```bash
cd /public/home/xdzs2026_c415

# 7. vLLM 编译 + 安装（容器重启后重做 install）
git clone -b v0.18.1 --depth 1 http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git
cd vllm_cscc
python setup.py bdist_wheel
cd dist
pip install vllm-*.whl --no-deps

# 8. 模型（断点续传：同一路径重跑即可）
cd /public/home/xdzs2026_c415
pip install modelscope
modelscope download --model Qwen/Qwen3.5-27B --local_dir ./Qwen3.5-27B
cp -r ./Qwen3.5-27B/ /root/Qwen3.5-27B

# 9. testdata
curl -f -C - -o testdata.tar.gz \
  https://zzefile.scnet.cn:65011/efile/s/d/c2N5MTE1OTkxMDU1OQ==/a927e65672549b46
mkdir -p ./testdata
tar -xzf testdata.tar.gz -C ./testdata --strip-components=1
chmod +x ./testdata/*.sh
```

检查：

```bash
du -sh ./Qwen3.5-27B          # 应 ~50G+
ls ./Qwen3.5-27B/config.json
ls ./testdata/start_vllm.sh
```

---

## 二、跑官方 baseline（PDF 第 10–11 步）

**终端 1：**

```bash
cd /public/home/xdzs2026_c415/testdata
./start_vllm.sh
```

**终端 2（等服务起来，curl 有 JSON 再测）：**

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-27B","messages":[{"role":"user","content":"你好，简单回复一句话。"}],"temperature":0.0,"max_tokens":64}'

cd /public/home/xdzs2026_c415/testdata
./run_throughput.sh 4-8K 10
./run_throughput.sh 8-16K 10
./run_throughput.sh 16-32K 10
./run_accuracy.sh hotpotqa 10
```

记录每档：**Output throughput、TTFT P99、TPOT P99** → 填 `report.md`

---

## 三、跑 lutinayi_branch 优化版

### 3.1 把代码弄进容器（只需一次）

```bash
cd /public/home/xdzs2026_c415
git clone -b lutinayi_branch \
  https://gitlab.eduxiji.net/fudiankuangxiangqu/2025pra-fdu-fudiankuangxiangqu.git
cd 2025pra-fdu-fudiankuangxiangqu
pip install -r requirements.txt
```

### 3.2 启动优化服务

**先停掉 baseline 的 start_vllm（若在跑）：** `pkill -f vllm` 或关掉终端 1

**终端 1：**

```bash
export MODEL_PATH=/root/Qwen3.5-27B
export PROJ=/public/home/xdzs2026_c415/2025pra-fdu-fudiankuangxiangqu
bash $PROJ/scripts/scnet_start_optimized.sh
```

### 3.3 测评

**终端 2：**

```bash
cd /public/home/xdzs2026_c415/testdata
./run_throughput.sh 4-8K 10
./run_throughput.sh 8-16K 10
./run_throughput.sh 16-32K 10

# Phase1 门禁（主攻 8-16K）
bash /public/home/xdzs2026_c415/2025pra-fdu-fudiankuangxiangqu/scripts/gate_check.sh quick
```

对比 baseline 与优化版三档数字，填 `report.md`。

---

## 四、lutinayi_branch 里有什么

| 文件 | 作用 |
|------|------|
| `launch.sh` | 评测机启动入口（warmup + prefix cache + fdu_vllm） |
| `scripts/scnet_start_optimized.sh` | SCNet 一键启动，端口 8001 |
| `scripts/gate_check.sh` | 吞吐+精度门禁 |
| `scripts/warmup_server.py` | 分档 warmup |
| `src/fdu_vllm/` | 优化插件（KV/GQA/可选 FP8） |
| `docs/easy_scoring.md` | 提分优先级 |

---

## 五、常见问题

| 问题 | 处理 |
|------|------|
| `Permission denied` on `*.sh` | `chmod +x testdata/*.sh` |
| curl 8001 refused | `start_vllm.sh` 没跑完；查 `cp /root/Qwen3.5-27B` |
| 10/10 failed | 服务未就绪；先 curl 成功再 throughput |
| 模型 38G | 未下完；`pip install modelscope` 后续传 |
| `/root/Qwen3.5-27B` 不存在 | 执行 `cp -r ./Qwen3.5-27B/ /root/Qwen3.5-27B` |

---

## 六、容器重启后最少重做

```bash
cd /public/home/xdzs2026_c415/vllm_cscc/dist && pip install vllm-*.whl --no-deps
cp -r ./Qwen3.5-27B/ /root/Qwen3.5-27B
# 然后启动 baseline 或 optimized
```
