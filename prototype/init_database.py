"""Initialize the local SQLite inspection-history database."""

from pathlib import Path

try:
    from .database import DEFAULT_DATABASE_PATH, initialize_database
except ImportError:
    from database import DEFAULT_DATABASE_PATH, initialize_database


def main() -> None:
    initialize_database(DEFAULT_DATABASE_PATH)
    print("Inspection database initialized.")
    print(f"Database path: {Path(DEFAULT_DATABASE_PATH).resolve()}")
    print("Tables: inspections, detections")
    print("The database stores inspection records and file paths—not model weights or training images.")


if __name__ == "__main__":
    main()
