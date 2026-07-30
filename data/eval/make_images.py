"""Generate the small sample images used by the eval set.

Run once before evaluating: ``python data/eval/make_images.py``. Images are
written to ``data/eval/images/`` and are deliberately not committed.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

IMAGES_DIR = Path(__file__).parent / "images"


def _new_canvas(width: int = 480, height: int = 260) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def make_code_image(path: Path) -> None:
    image, draw = _new_canvas()
    lines = [
        "def add(a, b):",
        "    result = a + b",
        "    return result",
        "",
        "print(add(2, 3))",
    ]
    draw.rectangle([10, 10, 470, 250], outline="black")
    for index, line in enumerate(lines):
        draw.text((24, 28 + index * 34), line, fill="black")
    image.save(path)


def make_chart_image(path: Path) -> None:
    image, draw = _new_canvas()
    draw.text((24, 16), "Requests per service", fill="black")
    bars = [("A", 60), ("B", 120), ("C", 90), ("D", 150)]
    base_y = 230
    for index, (label, value) in enumerate(bars):
        x0 = 40 + index * 100
        draw.rectangle([x0, base_y - value, x0 + 60, base_y], fill="black")
        draw.text((x0 + 24, base_y + 6), label, fill="black")
    image.save(path)


def make_error_image(path: Path) -> None:
    image, draw = _new_canvas()
    lines = [
        "Traceback (most recent call last):",
        '  File "app.py", line 1, in <module>',
        "    import requests",
        "ModuleNotFoundError: No module named",
        "'requests'",
    ]
    for index, line in enumerate(lines):
        draw.text((20, 24 + index * 34), line, fill="black")
    image.save(path)


def _arrow(draw: ImageDraw.ImageDraw, x0: int, y: int, x1: int) -> None:
    draw.line([x0, y, x1, y], fill="black", width=2)
    draw.polygon([(x1, y), (x1 - 8, y - 5), (x1 - 8, y + 5)], fill="black")


def make_diagram_image(path: Path) -> None:
    image, draw = _new_canvas()
    draw.text((20, 16), "Pipeline", fill="black")
    boxes = [("Input", 30), ("Agent", 195), ("Output", 360)]
    y = 110
    for label, x in boxes:
        draw.rectangle([x, y, x + 90, y + 50], outline="black", width=2)
        draw.text((x + 22, y + 18), label, fill="black")
    _arrow(draw, 122, y + 25, 193)
    _arrow(draw, 287, y + 25, 358)
    image.save(path)


def make_table_image(path: Path) -> None:
    image, draw = _new_canvas()
    draw.text((20, 16), "Scores", fill="black")
    rows = [("Name", "Score"), ("Ann", "90"), ("Bob", "75"), ("Cy", "82")]
    x0, y0, col_w, row_h = 40, 50, 160, 40
    for r, (left, right) in enumerate(rows):
        top = y0 + r * row_h
        draw.rectangle([x0, top, x0 + col_w, top + row_h], outline="black")
        draw.rectangle([x0 + col_w, top, x0 + 2 * col_w, top + row_h], outline="black")
        draw.text((x0 + 12, top + 12), left, fill="black")
        draw.text((x0 + col_w + 12, top + 12), right, fill="black")
    image.save(path)


def make_house_image(path: Path) -> None:
    image, draw = _new_canvas()
    draw.rectangle([180, 120, 300, 220], outline="black", width=2)  # body
    draw.polygon([(170, 120), (310, 120), (240, 60)], outline="black")  # roof
    draw.rectangle([225, 175, 255, 220], outline="black")  # door
    draw.rectangle([192, 138, 214, 160], outline="black")  # window
    image.save(path)


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    make_code_image(IMAGES_DIR / "code.png")
    make_chart_image(IMAGES_DIR / "chart.png")
    make_error_image(IMAGES_DIR / "error.png")
    make_diagram_image(IMAGES_DIR / "diagram.png")
    make_table_image(IMAGES_DIR / "table.png")
    make_house_image(IMAGES_DIR / "house.png")
    print(f"wrote sample images to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
