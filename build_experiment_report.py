"""Build a Chinese Word report summarizing the single-card MoE experiments."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "report_assets"
OUTPUT = ROOT / "实验结果总结报告_简明专业版.docx"
ASSET_DIR.mkdir(exist_ok=True)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_borders(cell, color: str = "D9D9D9", size: str = "4") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_cell_text(cell, text: str, bold: bool = False, color: str = "000000", size: int = 9) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_borders(cell)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, bold=True, color="FFFFFF", size=8)
        set_cell_shading(table.rows[0].cells[index], "1F4E78")
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for index, value in enumerate(row):
            set_cell_text(cells[index], value, size=8)
            if row_index % 2 == 1:
                set_cell_shading(cells[index], "F2F6FA")
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Cm(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(89, 89, 89)


def add_figure(doc: Document, path: Path, caption: str, width: float = 6.3) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    add_caption(doc, caption)


def add_bullet(doc: Document, text: str, level: int = 0) -> None:
    paragraph = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.add_run(text)


FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/calibri.ttf"),
]


def font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def canvas(title: str, width: int = 1200, height: int = 560):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 18), title, fill="#1F1F1F", anchor="ma", font=font(24))
    return image, draw


def axes(draw, left=90, top=75, right=1140, bottom=475, ymax=100, ylabel=""):
    draw.line((left, top, left, bottom), fill="#555555", width=2)
    draw.line((left, bottom, right, bottom), fill="#555555", width=2)
    for value in (0, ymax / 4, ymax / 2, ymax * 3 / 4, ymax):
        y = bottom - (value / ymax) * (bottom - top)
        draw.line((left, y, right, y), fill="#E6E6E6", width=1)
        draw.text((left - 12, y), f"{value:g}", fill="#444444", anchor="rm", font=font(15))
    if ylabel:
        draw.text((55, (top + bottom) // 2), ylabel, fill="#444444", anchor="mm", font=font(16))
    return left, top, right, bottom


def legend(draw, items, x=850, y=85):
    for index, (name, color) in enumerate(items):
        yy = y + index * 28
        draw.rectangle((x, yy - 7, x + 16, yy + 9), fill=color)
        draw.text((x + 24, yy), name, fill="#333333", anchor="lm", font=font(15))


def line_plot(draw, bounds, xs, ys, color, xlabels, ymin, ymax, show_points=True):
    left, top, right, bottom = bounds
    points = []
    for index, value in enumerate(ys):
        x = left + (right - left) * index / max(len(xs) - 1, 1)
        y = bottom - (value - ymin) / max(ymax - ymin, 1e-9) * (bottom - top)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=color, width=4)
    if show_points:
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    for point, label in zip(points, xlabels):
        draw.text((point[0], bottom + 20), str(label), fill="#444444", anchor="ma", font=font(14))


def bar_plot(draw, bounds, labels, values, colors, ymax):
    left, top, right, bottom = bounds
    slot = (right - left) / len(labels)
    bar_width = slot * 0.56
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        center = left + slot * (index + 0.5)
        x0 = center - bar_width / 2
        x1 = center + bar_width / 2
        y = bottom - value / ymax * (bottom - top)
        draw.rectangle((x0, y, x1, bottom), fill=color)
        draw.text((center, y - 10), f"{value:.2f}", fill="#333333", anchor="ms", font=font(14))
        draw.text((center, bottom + 20), label, fill="#444444", anchor="ma", font=font(13))


def chart_strategy(path: Path) -> None:
    image, draw = canvas("Experiment 1  Routing strategy comparison", 1400, 620)
    colors = ["#9EADBA", "#E39A54", "#4C9F70", "#2B6CB0"]
    labels = ["baseline", "token_drop", "expanded_drop", "priority_backtrack"]
    bounds = axes(draw, 90, 90, 650, 510, 110, "retention (%)")
    bar_plot(draw, bounds, labels, [100.00, 74.06, 99.53, 99.69], colors, 110)
    bounds = axes(draw, 760, 90, 1330, 510, 2.25, "")
    draw.text((720, 300), "imbalance", fill="#444444", anchor="mm", font=font(16))
    bar_plot(draw, bounds, labels, [2.000, 1.351, 1.005, 1.003], colors, 2.25)
    image.save(path)


def chart_capacity(path: Path) -> None:
    image, draw = canvas("Experiment 2  Capacity factor scan")
    bounds = axes(draw, 95, 85, 1135, 470, 105, "retention (%)")
    gamma = [0.5, 0.8, 1.0, 1.5]
    for name, data, color in zip(
        ["token_drop", "expanded_drop", "priority_backtrack"],
        [[47.66, 66.88, 74.06, 87.50], [50.00, 81.25, 99.53, 100.00], [50.00, 81.25, 99.69, 100.00]],
        ["#E39A54", "#4C9F70", "#2B6CB0"],
    ):
        line_plot(draw, bounds, gamma, data, color, gamma, 35, 105)
    legend(draw, [("token_drop", "#E39A54"), ("expanded_drop", "#4C9F70"), ("priority_backtrack", "#2B6CB0")])
    draw.text((660, 545), "capacity factor gamma", fill="#444444", anchor="ma", font=font(16))
    image.save(path)


def chart_topk(path: Path) -> None:
    image, draw = canvas("Experiment 7  top-k scan")
    bounds = axes(draw, 95, 85, 1135, 470, 105, "retention (%)")
    top_k = [1, 2, 3, 4]
    for name, data, color in zip(
        ["token_drop", "expanded_drop", "priority_backtrack"],
        [[25.00, 74.06, 90.62, 100.00], [100.00, 99.53, 98.44, 100.00], [100.00, 99.69, 100.00, 100.00]],
        ["#E39A54", "#4C9F70", "#2B6CB0"],
    ):
        line_plot(draw, bounds, top_k, data, color, top_k, 0, 105)
    legend(draw, [("token_drop", "#E39A54"), ("expanded_drop", "#4C9F70"), ("priority_backtrack", "#2B6CB0")])
    draw.text((660, 545), "top_k", fill="#444444", anchor="ma", font=font(16))
    image.save(path)


def chart_tp_ep(path: Path) -> None:
    image, draw = canvas("Experiment 6  TP and EP cost-model comparison", 1400, 620)
    batch = [1, 2, 4, 8, 16, 32, 64, 128]
    scenarios = [
        ("Hotspot load", [0.065, 0.130, 0.260, 0.520, 1.040, 2.080, 4.160, 8.320], [0.254, 0.588, 0.876, 1.335, 2.407, 4.640, 9.008, 17.662], 90, 650),
        ("Balanced load", [0.065, 0.130, 0.260, 0.520, 1.040, 2.080, 4.160, 8.320], [0.254, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096, 8.192], 760, 1320),
    ]
    for title, tp, ep, left, right in scenarios:
        draw.text(((left + right) // 2, 70), title, fill="#222222", anchor="ma", font=font(18))
        bounds = axes(draw, left, 95, right, 510, 18, "ms" if left == 90 else "")
        line_plot(draw, bounds, batch, tp, "#2B6CB0", batch, 0, 18)
        line_plot(draw, bounds, batch, ep, "#E39A54", batch, 0, 18)
    legend(draw, [("TP", "#2B6CB0"), ("EP", "#E39A54")], x=1140, y=120)
    draw.text((700, 560), "batch size", fill="#444444", anchor="ma", font=font(16))
    image.save(path)


def chart_counter(path: Path) -> None:
    image, draw = canvas("Experiment 8  Hysteresis counter width", 1200, 560)
    bounds = axes(draw, 90, 90, 510, 460, 2.6, "switches")
    bar_plot(draw, bounds, ["1", "2", "3", "4"], [2, 2, 1, 1], ["#6B8EAE"] * 4, 2.6)
    bounds = axes(draw, 650, 90, 1120, 460, 20, "")
    draw.text((610, 275), "total ms", fill="#444444", anchor="mm", font=font(16))
    line_plot(draw, bounds, [1, 2, 3, 4], [19.127, 18.396, 17.001, 17.001], "#2B6CB0", [1, 2, 3, 4], 16, 20)
    draw.text((300, 510), "counter bits", fill="#444444", anchor="ma", font=font(16))
    draw.text((880, 510), "counter bits", fill="#444444", anchor="ma", font=font(16))
    image.save(path)


def chart_overlap(path: Path) -> None:
    image, draw = canvas("Experiment 9  Compute and transfer overlap", 1200, 560)
    bounds = axes(draw, 100, 90, 1120, 470, 2.8, "stage ms")
    labels = ["1 missing", "2 missing", "4 missing"]
    serial = [2.176, 2.304, 2.560]
    overlap = [2.068, 2.068, 2.068]
    slot = (bounds[2] - bounds[0]) / len(labels)
    width = slot * 0.26
    for index, label in enumerate(labels):
        center = bounds[0] + slot * (index + 0.5)
        for offset, value, color in [(-width / 2, serial[index], "#C97A6B"), (width / 2, overlap[index], "#4C9F70")]:
            x0 = center + offset - width / 2
            x1 = center + offset + width / 2
            y = bounds[3] - value / 2.8 * (bounds[3] - bounds[1])
            draw.rectangle((x0, y, x1, bounds[3]), fill=color)
            draw.text(((x0 + x1) / 2, y - 10), f"{value:.3f}", fill="#333333", anchor="ms", font=font(13))
        draw.text((center, bounds[3] + 20), label, fill="#444444", anchor="ma", font=font(14))
    legend(draw, [("Serial", "#C97A6B"), ("Overlap", "#4C9F70")], x=930, y=105)
    image.save(path)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.1)
    section.right_margin = Cm(2.1)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.18
    for style_name, size in (("Title", 20), ("Heading 1", 15), ("Heading 2", 12)):
        style = styles[style_name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.size = Pt(size)
        style.font.bold = True
    styles["Title"].paragraph_format.space_after = Pt(8)
    styles["Heading 1"].paragraph_format.space_before = Pt(14)
    styles["Heading 1"].paragraph_format.space_after = Pt(6)
    styles["Heading 2"].paragraph_format.space_before = Pt(8)
    styles["Heading 2"].paragraph_format.space_after = Pt(4)


def build_doc() -> None:
    chart_strategy(ASSET_DIR / "strategy_comparison.png")
    chart_capacity(ASSET_DIR / "capacity_scan.png")
    chart_topk(ASSET_DIR / "topk_scan.png")
    chart_tp_ep(ASSET_DIR / "tp_ep_scan.png")
    chart_counter(ASSET_DIR / "counter_scan.png")
    chart_overlap(ASSET_DIR / "overlap_cost.png")

    doc = Document()
    configure_document(doc)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("单卡 MoE 实验结果总结 简明专业版")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("实验数据 图表和主要结论").italic = True
    subtitle.runs[0].font.color.rgb = RGBColor(89, 89, 89)
    subtitle.runs[0].font.size = Pt(11)
    doc.add_paragraph()

    p = doc.add_paragraph()
    p.add_run("先说结论。 ").bold = True
    p.add_run(
        "本报告总结单卡 MoE 模拟中的 11 项实验。实验结果表明，容量限制可以降低热点专家的过载，"
        "Expanded Drop 和 Priority Backtracking 通常能够在较少丢弃路由的情况下改善负载均衡。"
        "Priority Backtracking 属于局部启发式方法，不保证得到全局最优结果。TP 和 EP 的相对效果同时受到 batch size 和专家负载分布的影响。"
    )
    p = doc.add_paragraph()
    p.add_run("阅读方式。 ").bold = True
    p.add_run(
        "第 2 节给出实验清单，第 3 到第 6 节给出图表和结果解释。报告中的延迟来自教学成本模型，不是真实硬件测量。"
    )

    doc.add_heading("1 实验范围和统一配置", level=1)
    doc.add_paragraph(
        "本文保留 MoE 领域的必要术语。token 指模型输入中的一个处理单元；expert 指 MoE 中的子网络；"
        "Router 根据得分为 token 选择专家；capacity 指单个专家在当前批次中允许接收的最大路由数；"
        "TP 将同一层的计算拆分到多个设备，EP 将不同专家分布到不同设备。"
    )
    add_table(doc, ["项目", "设置"], [
        ["运行方式", "单卡纯 Python 逻辑模拟，不需要 CUDA"],
        ["逻辑专家数", "4"],
        ["token 数", "主要实验为 32；batch 扫描为 1 到 128"],
        ["top_k", "扫描实验为 1、2、3、4；其他实验通常为 2"],
        ["容量公式", "C = ceil(capacity_factor × T × top_k ÷ E)"],
        ["路由策略", "baseline、token_drop、expanded_drop、priority_backtrack"],
        ["记录方式", "每次运行追加到 research_log.md"],
    ], widths=[4.0, 11.5])
    doc.add_paragraph(
        "项目将真实系统简化为 Router 选择、容量控制、专家负载统计和 TP/EP 成本估算四个部分。"
        "实验用于验证算法趋势和参数影响，不用于复现特定 GPU 的真实性能。"
    )

    doc.add_heading("2 全部实验结果总览", level=1)
    add_table(doc, ["编号", "实验", "核心指标", "得到的主要认识"], [
        ["1", "四种路由策略", "保留率、最大负载、不均衡比", "容量控制降低热点专家的负载"],
        ["2", "capacity_factor 扫描", "容量因子与保留率/均衡", "容量增大通常减少路由丢弃，但不保证均衡"],
        ["3", "热点专家压力", "hot_bias 与路由结果", "分数增大不一定改变 Top-K 排名"],
        ["4", "候选专家冲突", "第一候选覆盖率、候选排名", "局部回溯有条件有效，不是全局最优"],
        ["5", "预留与回溯消融", "reservation_fraction、backtrack_depth", "参数依赖场景，深度超过 1 的收益有限"],
        ["6", "batch size 与 TP/EP", "两种模式的估算延迟", "热点负载偏向 TP，均衡负载偏向 EP"],
        ["7", "top_k 扫描", "top_k 与保留率/均衡", "候选数增加会增加路由压力，也可能改善补位"],
        ["8", "n 位计数器", "切换次数、总延迟", "位数越大越稳定，但响应更慢"],
        ["9", "计算与权重搬运重叠", "串行/重叠阶段耗时", "带宽足够时可隐藏部分搬运成本"],
    ], widths=[1.0, 4.0, 4.2, 6.3])

    doc.add_heading("3 路由策略和容量实验", level=1)
    doc.add_heading("3.1 四种路由策略比较", level=2)
    add_figure(doc, ASSET_DIR / "strategy_comparison.png", "图 1  实验 1 的路由保留率和专家负载不均衡比")
    doc.add_paragraph(
        "baseline 不设置容量上限，64 条路由全部保留，但 Expert 0 的负载达到 32，明显高于其他专家。"
        "token_drop 将每个专家的负载限制在容量 16 以内，但保留率下降到 74.06%。"
        "expanded_drop 会从候选专家中寻找仍有容量的专家，priority_backtrack 还会尝试重新安排部分已分配路由。"
        "两者的保留率约为 99.5% 到 99.7%，专家负载也接近均衡。"
    )
    doc.add_paragraph(
        "结果说明，容量感知路由需要同时考虑两点：减少路由丢弃，以及限制专家过载。baseline 的保留率最高，"
        "但没有遵守容量约束，因此不能仅凭保留率判断策略优劣。"
    )
    doc.add_paragraph(
        "这些数字不代表模型准确率，也不代表真实多 GPU 通信时间。它们只反映当前合成 Router 分布下的路由和负载变化。"
    )

    doc.add_heading("3.2 capacity_factor 扫描", level=2)
    add_figure(doc, ASSET_DIR / "capacity_scan.png", "图 2  实验 2 的容量因子扫描")
    doc.add_paragraph(
        "capacity_factor 从 0.5 增加到 1.5 时，容量感知策略的路由保留率总体上升。γ=0.5 时保留率约为 50%，"
        "γ=1.0 时 expanded_drop 和 priority_backtrack 已接近 100%。γ=1.5 时不均衡比达到 1.500，"
        "说明容量放宽后部分专家可能重新承担更多负载。"
    )
    doc.add_paragraph(
        "因此 capacity_factor 需要在路由保留率和负载均衡之间折中，不能简单认为越大越好。实际取值还要结合显存、计算能力和可接受的路由丢弃量。"
    )

    doc.add_heading("3.3 top_k 扫描", level=2)
    add_figure(doc, ASSET_DIR / "topk_scan.png", "图 3  实验 7 的 top-k 扫描")
    doc.add_paragraph(
        "top_k 表示每个 token 选择的专家数量。top_k 增大后，总路由数增加，但可用候选专家也更多。"
        "在本实验中，token_drop 在 top_k=1 时只能保留 25%；expanded_drop 和 priority_backtrack 可以利用其他候选专家补位。"
        "top_k=4 时每个 token 都可以选择 4 个专家，所有策略都达到 100% 保留和均衡。"
    )
    doc.add_paragraph(
        "top_k 增大同时带来更多路由计算和更多冲突处理工作。最终效果取决于候选专家的分布，不能只根据 top_k 判断。"
    )

    doc.add_heading("4 冲突处理和启发式参数", level=1)
    doc.add_paragraph(
        "实验 4 构造了三种候选专家冲突场景。Expanded Drop 和 Priority Backtracking 经常保留相同数量的路由，"
        "负载也同样均衡；但在 crossed_primary 和 scarce_backup 中，Priority Backtracking 的第一候选覆盖率更低。"
    )
    doc.add_paragraph(
        "这是局部启发式方法的限制。Priority Backtracking 只在有限范围内重新分配路由，"
        "不会搜索所有 token 的全局最优组合。因此，使用回溯不等于一定得到最优结果。"
    )
    doc.add_paragraph(
        "实验 5 扫描了 reservation_fraction 和 backtrack_depth。当前三种冲突场景中，"
        "backtrack_depth 从 1 增加到 2、3 几乎没有额外收益；过大的 reservation_fraction 还可能降低第一候选覆盖率。"
        "这说明两个参数需要根据路由分布调整。"
    )
    add_table(doc, ["指标", "Expanded Drop", "Priority Backtracking", "解释"], [
        ["shared_fallback 保留率", "32/48", "32/48", "两者相同"],
        ["crossed_primary 第一候选覆盖率", "66.7%", "41.7%", "局部回溯在此场景牺牲了更多第一候选"],
        ["scarce_backup 第一候选覆盖率", "66.7%", "41.7%", "备用专家稀缺时更明显"],
        ["三类场景负载", "均衡", "均衡", "两者都能控制专家容量"],
    ], widths=[4.8, 3.2, 4.2, 3.3])

    doc.add_heading("5 TP EP 选择和动态切换", level=1)
    add_figure(doc, ASSET_DIR / "tp_ep_scan.png", "图 4  实验 6 的 TP 和 EP 延迟成本模型")
    doc.add_paragraph(
        "左图是热点负载：大量路由集中到少数专家，EP 受到最大专家负载和不均衡惩罚的影响，batch 越大，EP 与 TP 的差距越明显。"
        "右图是均衡负载：batch 从 2 开始，EP 的估算延迟略低于 TP。"
        "因此 TP/EP 的选择需要同时考虑 batch size 和专家负载分布。"
    )
    add_figure(doc, ASSET_DIR / "counter_scan.png", "图 5  实验 8 的 n 位滞回计数器比较")
    doc.add_paragraph(
        "计数器根据连续多个时间段的请求率变化更新状态，达到上限或下限时才切换模式。"
        "1 位和 2 位计数器在本请求序列中切换了 2 次，3 位和 4 位切换了 1 次。位数越多，抗短时波动能力越强，但模式切换响应越慢。"
    )
    doc.add_paragraph(
        "综合实验中，请求率升高到 40 时从 TP 切换到 EP，降低到 2 时从 EP 切换回 TP。"
        "切换过程本身有额外成本，因此策略评价需要同时考虑稳态延迟和切换延迟。"
    )

    doc.add_heading("6 计算与专家权重搬运重叠", level=1)
    add_figure(doc, ASSET_DIR / "overlap_cost.png", "图 6  实验 9 的串行与重叠成本比较")
    doc.add_paragraph(
        "实验比较串行执行和重叠执行两种情况：串行执行先搬运专家权重，再进行计算；重叠执行同时进行计算和权重搬运。"
        "重叠模型的阶段耗时近似为计算时间和搬运时间中的较大值，再加同步开销。"
        "在均衡 EP、32 Gbps 链路和 512 MB 专家权重条件下，搬运 1、2、4 个缺失专家时，串行耗时为 2.176、2.304、2.560 ms，"
        "重叠耗时约为 2.068 ms。"
    )
    doc.add_paragraph(
        "当搬运时间小于计算时间时，重叠执行可以隐藏大部分搬运成本；当权重更大、缺失专家更多或链路更慢时，"
        "搬运时间可能超过计算时间，重叠不能完全消除额外开销。"
    )

    doc.add_heading("7 综合结论", level=1)
    add_bullet(doc, "容量感知路由首先用于限制热点专家的最大负载，其次才是提高路由保留率。")
    add_bullet(doc, "Expanded Drop 和 Priority Backtracking 在当前场景中保留率较高且负载均衡，但 Priority Backtracking 不是全局最优算法。")
    add_bullet(doc, "capacity_factor、top_k、reservation_fraction 和 backtrack_depth 需要联合调节。")
    add_bullet(doc, "热点负载或小 batch 更适合 TP；均衡的大 batch 更适合 EP。")
    add_bullet(doc, "计数器位数增加可以减少频繁切换，但会降低对负载变化的响应速度。")
    add_bullet(doc, "计算和权重搬运可以并行时，足够的链路带宽能够降低搬运的可见开销。")

    doc.add_heading("8 实验边界和下一步", level=1)
    doc.add_paragraph(
        "本报告验证的是算法机制和趋势，不是论文的真实多 GPU 性能复现。当前项目没有执行真实 CUDA kernel、"
        "All-to-All、All-Reduce、NVLink、InfiniBand，也没有加载真实语言模型或测量 MMLU、GSM8K 等任务准确率。"
    )
    doc.add_paragraph(
        "如果继续接近真实系统，可以将逻辑专家替换为小型 PyTorch MoE，在 GPU 上执行真实 tensor 操作并用 CUDA event 记录时间，"
        "然后加入多 GPU 通信和专家权重缓存。当前报告适合作为论文理解和后续实验的基础。"
    )

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_doc()
