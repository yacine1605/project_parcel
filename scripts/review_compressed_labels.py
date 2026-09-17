"""Interactive, resumable review of compressed annotations.

This tool edits only the review checklist. It never writes to the dataset.
Keyboard shortcuts: C keep compressed, H hole, M minor damage, W wet, X exclude,
N needs domain review, Left/Right navigate.
"""

from __future__ import annotations

import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk


ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = ROOT / "reports" / "compressed_ontology_review" / "compressed_review_checklist.csv"
CLASS_NAMES = ["minor_damage", "compressed", "hole", "wet"]
COLORS = ["#3b82f6", "#22c55e", "#ef4444", "#eab308"]
DECISIONS = [
    ("Keep compressed", "keep_compressed", "c"),
    ("Relabel hole", "relabel_hole", "h"),
    ("Relabel minor", "relabel_minor_damage", "m"),
    ("Relabel wet", "relabel_wet", "w"),
    ("Exclude", "exclude_unverifiable", "x"),
    ("Needs review", "needs_domain_review", "n"),
]


class ReviewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.rows = self.load_rows()
        self.fieldnames = list(self.rows[0])
        self.index = next((i for i, row in enumerate(self.rows) if not row["review_decision"].strip()), 0)
        self.photo = None

        root.title("Compressed Ontology Review — V2 read-only")
        root.geometry("1180x860")
        root.minsize(900, 700)

        self.header = ttk.Label(root, font=("Segoe UI", 11, "bold"))
        self.header.pack(fill="x", padx=12, pady=(10, 4))
        self.canvas = tk.Canvas(root, bg="#171717", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=4)

        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=12, pady=6)
        for label, decision, key in DECISIONS:
            ttk.Button(controls, text=f"{label} [{key.upper()}]", command=lambda d=decision: self.decide(d)).pack(
                side="left", padx=3
            )
        ttk.Button(controls, text="← Previous", command=lambda: self.move(-1)).pack(side="right", padx=3)
        ttk.Button(controls, text="Next →", command=lambda: self.move(1)).pack(side="right", padx=3)

        notes_frame = ttk.Frame(root)
        notes_frame.pack(fill="x", padx=12, pady=(0, 10))
        ttk.Label(notes_frame, text="Notes:").pack(side="left")
        self.notes = ttk.Entry(notes_frame)
        self.notes.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(notes_frame, text="Save notes", command=self.save_notes).pack(side="right")

        for _, decision, key in DECISIONS:
            root.bind(key, lambda event, d=decision: self.decide(d))
            root.bind(key.upper(), lambda event, d=decision: self.decide(d))
        root.bind("<Left>", lambda event: self.move(-1))
        root.bind("<Right>", lambda event: self.move(1))
        root.bind("<Configure>", lambda event: self.render())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.render()

    @staticmethod
    def load_rows() -> list[dict[str, str]]:
        with CHECKLIST.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def save(self) -> None:
        temporary = CHECKLIST.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(CHECKLIST)

    def read_labels(self, path: Path):
        labels = []
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                labels.append((int(fields[0]), *(float(value) for value in fields[1:])))
        return labels

    def annotated_image(self) -> Image.Image:
        row = self.rows[self.index]
        image = Image.open(ROOT / row["image"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for class_id, x, y, width, height in self.read_labels(ROOT / row["label"]):
            x1, y1 = (x - width / 2) * image.width, (y - height / 2) * image.height
            x2, y2 = (x + width / 2) * image.width, (y + height / 2) * image.height
            draw.rectangle((x1, y1, x2, y2), outline=COLORS[class_id], width=max(3, image.width // 300))
            draw.text((x1 + 3, max(0, y1 - 14)), CLASS_NAMES[class_id], fill=COLORS[class_id], font=font)
        return image

    def render(self) -> None:
        if not self.rows or self.canvas.winfo_width() < 10:
            return
        row = self.rows[self.index]
        done = sum(bool(item["review_decision"].strip()) for item in self.rows)
        self.header.configure(
            text=(f"{self.index + 1}/{len(self.rows)} | completed {done}/{len(self.rows)} | "
                  f"split={row['split']} | flag={row.get('source_name_flag') or 'none'} | "
                  f"decision={row['review_decision'] or 'pending'}")
        )
        self.notes.delete(0, "end")
        self.notes.insert(0, row.get("review_notes", ""))
        image = self.annotated_image()
        image.thumbnail((self.canvas.winfo_width() - 20, self.canvas.winfo_height() - 20))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, image=self.photo)

    def decide(self, decision: str) -> None:
        self.rows[self.index]["review_notes"] = self.notes.get().strip()
        self.rows[self.index]["review_decision"] = decision
        self.save()
        self.move(1)

    def save_notes(self) -> None:
        self.rows[self.index]["review_notes"] = self.notes.get().strip()
        self.save()
        self.render()

    def move(self, delta: int) -> None:
        self.index = max(0, min(len(self.rows) - 1, self.index + delta))
        self.render()

    def close(self) -> None:
        self.rows[self.index]["review_notes"] = self.notes.get().strip()
        self.save()
        self.root.destroy()


def main() -> None:
    if not CHECKLIST.exists():
        raise FileNotFoundError(f"Generate the checklist first: {CHECKLIST}")
    root = tk.Tk()
    ReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
