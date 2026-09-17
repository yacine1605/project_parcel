# Phase 17 — SQLite Inspection Database

## Purpose

A database stores structured information so inspection results can be saved, searched, filtered, summarized, and displayed later. This SQLite database contains inspection records and paths to image files. It does **not** contain model weights, training data, or full image binary data.

SQLite is suitable for this certification prototype because it is local, uses one file, requires no database server, is supported by Python's standard `sqlite3` module, and integrates easily with a future Streamlit dashboard.

Database path:

```text
data/parcel_inspections.db
```

## Primary key versus package ID

`inspections.id` is an automatically increasing integer **primary key**. A primary key uniquely identifies one database row and allows other tables to refer to it reliably.

`package_id` is different. It represents the warehouse's operational parcel or barcode identifier. During a real deployment it should come from the logistics system. When absent, the code generates an identifier such as:

```text
PROTO-20260828T223000Z-A1B2C3D4
```

That generated value is explicitly a prototype ID, not a real tracking number.

## Schema

### `inspections`

One row represents one completed two-model inspection.

| Field | Meaning |
|---|---|
| `id` | Internal auto-incrementing primary key |
| `package_id` | Barcode/operational ID or generated prototype ID |
| `timestamp_utc` | Time inference completed, in UTC |
| `image_path` | Original image file location |
| `annotated_image_path` | Output image containing evidence overlays |
| `classifier_label` | Frozen MobileNet `damaged` or `intact` result |
| `classifier_probability` | Damaged probability from sigmoid |
| `classifier_threshold` | Frozen decision threshold used |
| `yolo_detection_count` | Number of linked detector results |
| `prototype_decision` | Conservative `ACCEPT` or `REVIEW` status |
| `classifier_latency_ms` | MobileNet forward time recorded by the pipeline |
| `yolo_latency_ms` | YOLO inference component time |
| `total_latency_ms` | Recorded combined pipeline processing time |
| `classifier_model` | Model architecture name |
| `classifier_version` | Frozen classifier version |
| `yolo_model` | Detector architecture name |
| `yolo_version` | Completed detector experiment version |
| `created_at_utc` | Time the database row was written |

Probability, threshold, latency, count, label, and decision fields have SQL `CHECK` constraints. Required values are `NOT NULL`.

### `detections`

Each row represents one YOLO detection belonging to an inspection.

| Field | Meaning |
|---|---|
| `id` | Internal detection primary key |
| `inspection_id` | Foreign key linking to `inspections.id` |
| `class_name` | `minor_damage`, `compressed`, `hole`, or `wet` |
| `confidence` | Detector confidence, not severity |
| `x1`, `y1` | Left/top box coordinates |
| `x2`, `y2` | Right/bottom box coordinates |

Relationship:

```text
one inspection
      |
      +---- zero detections
      +---- one detection
      +---- many detections
```

This one-to-many design is preferable to stuffing several boxes, classes, and confidences into one text column. SQL can directly find all inspections containing `hole`, count damage classes, or retrieve every box for one parcel.

Foreign-key enforcement is enabled for every connection. An orphan detection referencing a nonexistent inspection is rejected. `ON DELETE CASCADE` ensures that if an inspection is deliberately deleted in a future maintenance workflow, its detections do not remain orphaned.

## Database operations

`prototype/database.py` provides small functions rather than an ORM:

- `initialize_database()` — creates tables and indexes;
- `save_inspection()` — validates and inserts an inspection plus detections;
- `get_recent_inspections()` — newest records, default limit 10;
- `get_inspection_by_id()` — one inspection with linked detections;
- `get_review_inspections()` — records requiring REVIEW;
- `get_inspections_by_damage_class()` — records containing a requested YOLO class;
- `get_daily_counts()` — UTC inspection counts by date;
- `get_basic_analytics()` — total, ACCEPT/REVIEW counts, average probability, average latency, and detection-class counts.

A **connection** opens a session with SQLite. A **cursor** executes SQL such as `CREATE TABLE`, `INSERT`, and `SELECT`. `commit()` permanently saves schema and row changes; `rollback()` cancels the current transaction after an error; `close()` releases the file.

Parameterized `?` placeholders pass package IDs and paths separately from SQL, avoiding unsafe string construction.

## Transaction integrity

`save_inspection()` validates the evidence before insertion. It then inserts the parent inspection and every child detection in one transaction:

```text
validate evidence
    |
INSERT inspection
    |
INSERT zero or more detections
    |
commit everything together
```

If any detection insertion fails, `rollback()` prevents a partially saved inspection. The stored detection count must equal the supplied detection list length, coordinates must form a valid `xyxy` box, and probabilities/confidences must remain between 0 and 1.

## Phase 16 integration

Persistence occurs after both independent model paths complete:

```text
image
  |
  +--> frozen MobileNet
  +--> selected YOLO
          |
          v
combined evidence + prototype decision
          |
          v
optional SQLite save
```

`inspection_pipeline.save_completed_inspection()` receives already-completed evidence. Database activity cannot alter model predictions. The single-image command now supports:

```powershell
.\venv\Scripts\python.exe -m prototype.single_image_demo parcel.jpg `
    --save-to-database `
    --package-id "WAREHOUSE-BARCODE-123"
```

If `--package-id` is omitted, a clearly labeled prototype identifier is generated.

Images are not inserted as BLOBs. Large image binaries would make SQLite unnecessarily heavy; original and annotated paths are stored instead.

## Saved development examples

Three genuine development/validation images were processed with the unchanged Phase 16 models and saved:

| Row ID | Package ID | MobileNet | Probability | YOLO detections | Decision | Total time |
|---:|---|---|---:|---|---|---:|
| 1 | `DEMO-HOLE-001` | damaged | 0.9869 | hole | REVIEW | 480.11 ms |
| 2 | `DEMO-COMPRESSED-001` | damaged | 0.9730 | minor_damage, compressed | REVIEW | 475.86 ms |
| 3 | `DEMO-WET-001` | damaged | 0.9859 | wet | REVIEW | 516.61 ms |

These are prototype development records, not production parcels or final-test samples. The detector's class prediction is preserved exactly even when it differs from the source image selection category.

Current analytics:

| Value | Result |
|---|---:|
| Total inspections | 3 |
| ACCEPT | 0 |
| REVIEW | 3 |
| Detection rows | 4 |
| Average classifier probability | 0.9820 |
| Average total pipeline latency | 490.86 ms |
| Detection-class counts | compressed 1, hole 1, minor_damage 1, wet 1 |

## Query examples

### Recent ten

```python
recent = get_recent_inspections(limit=10)
```

Equivalent SQL:

```sql
SELECT * FROM inspections
ORDER BY timestamp_utc DESC, id DESC
LIMIT 10;
```

### REVIEW queue

```sql
SELECT * FROM inspections
WHERE prototype_decision = 'REVIEW'
ORDER BY timestamp_utc DESC;
```

### Inspections containing a hole

```sql
SELECT DISTINCT i.*
FROM inspections AS i
JOIN detections AS d ON d.inspection_id = i.id
WHERE d.class_name = 'hole';
```

### Daily counts

```sql
SELECT substr(timestamp_utc, 1, 10) AS inspection_date,
       COUNT(*) AS inspection_count
FROM inspections
GROUP BY substr(timestamp_utc, 1, 10);
```

## Verification

`scripts/test_inspection_database.py` created a temporary database and verified:

- three inspections inserted and retrieved;
- every linked detection returned with its inspection;
- recent, REVIEW, class-filter, daily-count, and analytics queries;
- expected counts;
- foreign-key rejection of an orphan detection;
- transactional save behavior.

The permanent database was then queried independently. Foreign-key status returned enabled, three inspection rows were present, and all four detection rows were linked correctly.

## Limitations

- Package IDs in the saved examples are demonstration identifiers.
- SQLite is suitable for a local prototype, not necessarily a high-volume multi-site warehouse.
- File paths can become stale if images are moved outside a managed storage policy.
- The ACCEPT/REVIEW policy remains a prototype rule.
- No human ground-truth disposition, severity, reject decision, or correction workflow is stored yet.
- Concurrent writes and backup/retention policy require future deployment design.

## Artifacts

- `prototype/database.py`
- `prototype/init_database.py`
- `scripts/test_inspection_database.py`
- `data/parcel_inspections.db`
- `prototype_outputs/phase17/`

## Closure

No model was retrained or modified. MobileNet weights, threshold, preprocessing, YOLO checkpoint, class mapping, and all final results remain unchanged. Streamlit was not started.
