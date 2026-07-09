#!/usr/bin/env python3
"""Generate baseline_result.pdf from current known metrics."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "docs" / "baseline_result.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Baseline Result",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=16, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=4)
    note = ParagraphStyle("note", parent=styles["BodyText"], fontSize=8, textColor=colors.HexColor("#555555"))

    story = [
        Paragraph("Baseline 测评结果记录", title),
        Paragraph(
            f"团队：复旦大学 · 更新日期：{date.today().isoformat()} · "
            "赛题：基于国产加速卡的 Qwen 大模型推理服务优化（初赛）",
            body,
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("1. SCNet 自测（stock vLLM · start_vllm.sh）", h2),
        Paragraph(
            "<b>状态：未完成。</b> 2026-07-08 在 worker-0 容器尝试跑吞吐，"
            "10/10 请求全部失败（Output throughput = 0）。"
            "根因：vLLM 未在 8001 正常启动；模型仅 38G（4 个 safetensors，可能未下完）；"
            "/root/Qwen3.5-27B 不存在；modelscope 未安装。",
            body,
        ),
        Spacer(1, 0.15 * cm),
        _table(
            [
                ["档位", "权重", "Output tok/s", "TTFT P99", "TPOT P99", "SLA", "备注"],
                ["4-8K", "20%", "—", "—", "—", "—", "待 PDF 流程重跑"],
                ["8-16K", "50%", "—", "—", "—", "—", "待 PDF 流程重跑"],
                ["16-32K", "30%", "—", "—", "—", "—", "待 PDF 流程重跑"],
            ],
            col_widths=[2.0 * cm, 1.2 * cm, 2.2 * cm, 2.0 * cm, 2.0 * cm, 1.2 * cm, 4.0 * cm],
        ),
        Spacer(1, 0.25 * cm),
        Paragraph("2. 竞赛平台实测（排行榜 · 账号：富贵花开）", h2),
        Paragraph(
            "来源：希冀排行榜，最后提交 2026-07-06 09:54:48。"
            "平台仅展示三档实际吞吐，不含 TTFT/TPOT 明细；"
            "该得分为<strong>已提交优化方案</strong>的平台评测结果，非 SCNet stock baseline。",
            body,
        ),
        Spacer(1, 0.15 * cm),
        _table(
            [
                ["档位", "权重", "Output tok/s", "TTFT P99", "TPOT P99", "SLA扣分", "精度扣分"],
                ["4-8K", "20%", "18.37", "—", "—", "0", "0"],
                ["8-16K", "50%", "16.65", "—", "—", "0", "0"],
                ["16-32K", "30%", "13.49", "—", "—", "0", "0"],
                ["合计", "100%", "—", "—", "—", "0", "0"],
            ],
            col_widths=[2.0 * cm, 1.2 * cm, 2.2 * cm, 2.0 * cm, 2.0 * cm, 1.5 * cm, 1.5 * cm],
        ),
        Spacer(1, 0.1 * cm),
        Paragraph("最终得分：<b>84.7446</b> · 排名：<b>#26</b> / 107（2026-07-08 快照）", body),
        Spacer(1, 0.25 * cm),
        Paragraph("3. 复旦其他账号参考（平台）", h2),
        _table(
            [
                ["队伍", "8K-16K", "4K-8K", "16K-32K", "最终得分", "最后提交"],
                ["富贵花开", "16.65", "18.37", "13.49", "84.74", "2026-07-06"],
                ["LUX.", "13.96", "15.68", "10.02", "77.27", "2026-07-08"],
                ["啊对对对队", "10.04", "12.95", "5.76", "59.96", "2026-06-26"],
                ["AI说的不如绿猫说的队", "10.01", "12.87", "5.76", "59.82", "2026-07-08"],
            ],
            col_widths=[4.5 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.0 * cm, 2.5 * cm],
        ),
        Spacer(1, 0.25 * cm),
        Paragraph("4. 精度自测（SCNet · run_accuracy.sh）", h2),
        _table(
            [
                ["任务", "指标", "Baseline", "当前方案", "Δ", "状态"],
                ["hotpotqa（问答）", "F1", "—", "—", "—", "未跑通"],
                ["gov_report（摘要）", "ROUGE-L", "—", "—", "—", "未跑通"],
                ["retrieval_multi_point", "Accuracy", "—", "—", "—", "未跑通"],
                ["aggregation_keyword", "Accuracy", "—", "—", "—", "未跑通"],
            ],
            col_widths=[3.5 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.2 * cm, 2.5 * cm],
        ),
        Spacer(1, 0.25 * cm),
        Paragraph("5. 与榜首差距（豆包F4 · #1）", h2),
        _table(
            [
                ["档位", "榜首 tok/s", "我们 tok/s", "差距", "权重"],
                ["8K-16K", "19.51", "16.65", "-2.86", "50%"],
                ["4K-8K", "21.42", "18.37", "-3.05", "20%"],
                ["16K-32K", "15.05", "13.49", "-1.56", "30%"],
            ],
            col_widths=[2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm],
        ),
        Spacer(1, 0.25 * cm),
        Paragraph("6. 下一步（按 PDF 文档顺序）", h2),
        Paragraph(
            "① pip install modelscope → 续传 Qwen3.5-27B 至 50G+；"
            "② cp -r ~/Qwen3.5-27B /root/Qwen3.5-27B；"
            "③ cd ~/testdata && ./start_vllm.sh；"
            "④ curl 验证后 ./run_throughput.sh 三档；"
            "⑤ ./run_accuracy.sh 四类精度；"
            "⑥ 将 TTFT P99 / TPOT P99 回填本 PDF 与 report.md §4。",
            body,
        ),
        Spacer(1, 0.15 * cm),
        Paragraph(
            "说明：「—」表示尚无有效 SCNet 实测数据；平台吞吐可用于跟踪排名，"
            "不能替代官方 stock baseline 对照。",
            note,
        ),
    ]
    doc.build(story)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
