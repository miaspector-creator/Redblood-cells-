# TXL-PBC Dataset Report

Inspection date: 2026-11-02

## Scope

This report covers **only `Data/TXL-PBC.zip`**, as requested. The other dataset
archives are not part of this task.

## Archive status

The TXL-PBC archive is damaged and cannot be opened or extracted with the
available ZIP tools. Its recorded central-directory metadata is inconsistent
with the actual file size, and parts of its binary data appear to have been
transformed as UTF-8 replacement characters. The original archive has been
left untouched.

## Recoverable path-level evidence

A read-only scan of the damaged archive's raw bytes recovered these matching
image/label filename pairs:

| Split | Image paths | Matching label paths |
|---|---:|---:|
| Train | 882 | 882 |
| Validation | 252 | 252 |
| Test | 126 | 126 |
| **Total** | **1,260** | **1,260** |

Recovered paths indicate a layout resembling
`images/{train,val,test}/<id>.png` and `labels/{train,val,test}/<id>.txt`.
Paths for `data.yaml` and `README.md` are also visible in the archive bytes.
This is evidence of a YOLO-style dataset layout, **not verification that the
archive contains valid YOLO data**.

## Unknown / unverified

Because file payloads cannot be reliably read:

- Image decodability, dimensions, and visual contents are unknown.
- Label text syntax, class IDs, bounding-box values, and image/label semantic
  correctness are unknown.
- Class names and the contents of `data.yaml` and `README.md` are unknown.
- The detected path counts do not constitute a ZIP integrity check or data
  quality validation.

## Training readiness

Training is blocked until a clean copy of TXL-PBC is provided and its archive
integrity, annotation files, class manifest, and split contents are validated.
