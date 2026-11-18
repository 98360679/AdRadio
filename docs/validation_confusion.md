# Clean-classifier validation accuracy (baseline, pre-attack)

The victim CVNN's accuracy on the **held-out run-3 validation** (leakage-safe split: runs 1–2 train,
run 3 validate), classifying each 802.11 frame by **majority vote over its 13 I/Q frames**. This is the
clean baseline the adversarial results degrade from.

Regenerate from the shipped bundle (enrollment + `fingerprint_cvnn.pt`):
```bash
export CAPTURES_ROOT=/path/to/captures
python3 exp3_validation_confusion.py          # -> docs/validation_confusion.csv
```

**Frame-level accuracy: 95.98%** · per-I/Q-frame (window-level): 92.34% · over 523 frames.

## Frame-level confusion (majority vote)

| true＼pred | device_1 | device_2 | device_3 | device_4 | device_5 | n | acc |
|---|---|---|---|---|---|---|---|
| **device_1** | 105 | 0 | 0 | 0 | 0 | 105 | 100.0% |
| **device_2** | 0 | 87 | 15 | 4 | 0 | 106 | 82.1% |
| **device_3** | 0 | 0 | 101 | 2 | 0 | 103 | 98.1% |
| **device_4** | 0 | 0 | 0 | 105 | 0 | 105 | 100.0% |
| **device_5** | 0 | 0 | 0 | 0 | 104 | 104 | 100.0% |
| **overall** | | | | | | 523 | **95.98%** |

Four of five devices classify at 98–100%; the only material confusion is **device_2** (82.1%),
misread as device_3 (15 frames) or device_4 (4). The full counts (frame- and window-level) are in
`validation_confusion.csv`.

*Model `fingerprint_cvnn.pt` and the enrollment run-3 captures are byte-identical to the shipped bundle;
a reviewer reproduces these exact numbers by running the command above against it.*
