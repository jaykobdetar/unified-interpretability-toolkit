# Changelog

## Unreleased

- Preserve a single-tensor calibration worker's failure when it finishes immediately after queue admission. Queue admission and clearing the prior error now share the progress lock; refused requests retain that error. HTTP statuses, response wording and calibration formats are unchanged.
