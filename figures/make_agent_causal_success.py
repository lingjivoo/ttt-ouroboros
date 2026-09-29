#!/usr/bin/env python3
"""Draw the audited WebShop feedback comparison as a compact vector PDF."""

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig_agent_causal_success.pdf"

W, H = 220, 190
LEFT, RIGHT, BOTTOM, TOP = 36, 214, 49, 158
RED = HexColor("#AC3E2C")
BLUE = HexColor("#3D6997")
GREEN = HexColor("#4C7C57")
GRAY = HexColor("#86817A")
GRID = HexColor("#E7E4DF")
INK = HexColor("#252525")

labels = ["Writes Off", "Closed", "Fixed", "Settlement"]
values = [0.1600, 0.1053, 0.1800, 0.1853]
errors = [0.0, 0.0597, 0.0089, 0.0202]
colors = [GRAY, RED, BLUE, GREEN]
xs = [58, 103, 148, 195]


def ypos(value: float) -> float:
    return BOTTOM + value / 0.25 * (TOP - BOTTOM)


def errorbar(c, x, value, sd, color):
    yy, lo, hi = ypos(value), ypos(max(0, value - sd)), ypos(min(0.25, value + sd))
    c.setStrokeColor(color)
    c.setLineWidth(1.0)
    c.line(x, lo, x, hi)
    c.line(x - 3, lo, x + 3, lo)
    c.line(x - 3, hi, x + 3, hi)
    c.setFillColor(color)
    c.circle(x, yy, 3.1, stroke=0, fill=1)


c = canvas.Canvas(str(OUT), pagesize=(W, H))
c.setTitle("WebShop Held-Out Exact Success")

# Grid and axes.
for value in [0, 0.05, 0.10, 0.15, 0.20, 0.25]:
    y = ypos(value)
    c.setStrokeColor(GRID)
    c.setLineWidth(0.55)
    c.line(LEFT, y, RIGHT, y)
    c.setFillColor(INK)
    c.setFont("Times-Roman", 8)
    c.drawRightString(LEFT - 4, y - 2.5, f"{value:.2f}")
c.setStrokeColor(INK)
c.setLineWidth(0.7)
c.line(LEFT, BOTTOM, RIGHT, BOTTOM)
c.line(LEFT, BOTTOM, LEFT, TOP)

for x, label, value, sd, color in zip(xs, labels, values, errors, colors):
    errorbar(c, x, value, sd, color)
    c.setFillColor(INK)
    c.setFont("Times-Roman", 7.4)
    if label == "Writes Off":
        c.drawCentredString(x, BOTTOM - 12, "Writes")
        c.drawCentredString(x, BOTTOM - 21, "Off")
    else:
        c.drawCentredString(x, BOTTOM - 16, label)

c.saveState()
c.translate(10, (BOTTOM + TOP) / 2)
c.rotate(90)
c.setFont("Times-Roman", 8.5)
c.setFillColor(INK)
c.drawCentredString(0, 0, "Held-Out Exact Success")
c.restoreState()
c.setFont("Times-Roman", 9)
c.drawCentredString(W / 2, 8, "(c) WebShop")
c.save()
print(OUT)
