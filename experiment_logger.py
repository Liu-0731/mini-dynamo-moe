"""Append reproducible experiment results to a human-readable Markdown log."""

from datetime import datetime
from pathlib import Path


LOG_PATH = Path(__file__).with_name("research_log.md")


def append_experiment(
    title: str,
    question: str,
    setup: str,
    result_markdown: str,
    analysis: list[str],
    limitations: list[str],
) -> None:
    """Append one experiment entry without deleting previous entries."""
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\n## {title}\n\n")
        log.write(f"- 运行时间：`{timestamp}`\n")
        log.write(f"- 研究问题：{question}\n")
        log.write(f"- 固定条件：{setup}\n\n")
        log.write("### 结果\n\n")
        log.write(result_markdown.rstrip() + "\n\n")
        log.write("### 自动分析\n\n")
        for item in analysis:
            log.write(f"- {item}\n")
        log.write("\n### 解释边界\n\n")
        for item in limitations:
            log.write(f"- {item}\n")
        log.write("\n---\n")


if __name__ == "__main__":
    print(f"实验日志位置：{LOG_PATH}")
