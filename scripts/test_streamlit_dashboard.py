"""
# ============================================================
# What are we testing in this script?
# ============================================================
#
# This lightweight Phase 18 check opens every Streamlit page without starting a
# browser. It verifies that history, REVIEW, and analytics views can read the
# existing SQLite database. It does NOT train models or access test images.
#
# A real development-image inference check is documented in the Phase 18 report
# because model loading is intentionally not repeated in this fast UI smoke test.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "dashboard" / "app.py"


def main() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception, list(app.exception)
    assert app.title[0].value.endswith("Parcel Inspection")

    for page in ["Inspection History", "Review Queue", "Analytics"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"{page} failed: {list(app.exception)}"
        print(f"{page}: PASS")

    print("Parcel Inspection: PASS")
    print("Dashboard smoke test complete. No model was trained and no test image was used.")


if __name__ == "__main__":
    main()
