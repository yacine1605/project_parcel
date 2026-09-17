# Experiment report archive

The reports in this directory document the project chronologically. They are preserved to show how dataset policy, baselines, transfer learning, error analysis, architecture, and the user interface evolved.

They are not all descriptions of the current application.

For current runtime behavior, use:

1. the root `README.md` for setup and project orientation;
2. `docs/PARCEL_STATE_INFERENCE_ARCHITECTURE.md` for the authoritative inference contract;
3. `docs/PRESENTATION_GUIDE.md` for current claims and presentation language.

In particular, Phase 13, Phase 16, and Phase 18 describe the earlier two-model classifier-plus-detector prototype. The current application adds generic parcel localization and a dedicated open/closed crop classifier, and conditionally executes downstream damage models.

## Reading order for learners

- Dataset and ontology work: Phase 7 through Phase 8 and the V3 correction reports.
- Scratch CNN and transfer learning: Phase 9 through Phase 12.
- Historical architecture decision: Phase 13.
- Feasibility audits: Phase 14 and Phase 15.
- Historical prototype and persistence: Phase 16 through Phase 18.
- Robustness planning: Phase 19/20.

Reported metrics belong to the dataset and protocol named in each report. Do not combine component-level results into a single end-to-end accuracy claim.
