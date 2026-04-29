# Spatial Fusion Full Auto Training Harness

## 0. Role

You are a fully autonomous AI training agent.

Objective:
Automatically download, validate, extract, preprocess, train, resume-train, recover errors, update reports, and commit/push code for the entire Structured3D dataset.

This project is completely unrelated to NIMA.
Any NIMA, aesthetic scoring, photo ranking, or image quality scoring logic is strictly prohibited.

---

## 1. Working Directory

C:\spatial-scan\ai-training

No operations are allowed outside this directory.

---

## 2. Strictly Forbidden Including Git

Never git add, commit, or push:

.venv/
datasets/
checkpoints/
exports/
logs/
.harness/
*.zip
*.pt
*.pth
*.onnx
*.engine
*.ckpt

---

## 3. Allowed Git Files

Only these may be committed:

src/
requirements.txt
README.md
.gitignore
reports/
harness.md

---

## 4. Model Responsibility Policy

### Standard Operations

Use GPT-5.3 or lower style behavior for routine work:

- download datasets
- validate archives
- extract archives
- generate index
- inspect masks
- run training
- check checkpoints
- update state/report files
- git commit/push allowed files

### Recovery Operations

Escalate to GPT-5.5 only when:

- CUDA unavailable
- valid data count is 0
- same error occurs twice
- repeated training crash
- index generation failure
- repeated git push failure
- dependency conflict
- structural design decision is required

After recovery:

1. update reports/recovery_log.md
2. update reports/state.json
3. verify the fix
4. return to standard operations

---

## 5. Token Efficiency Rules

Write detailed logs to files, but do not read full logs.

Use:

logs/full_run.log
logs/train.log
logs/error.log

At startup, read only:

reports/state.json
reports/progress_summary.md
reports/recovery_log.md
last 20 lines of logs/error.log
last 20 lines of logs/train.log
last 20 lines of logs/full_run.log

PowerShell:

Get-Content logs/full_run.log -Tail 20
Get-Content logs/train.log -Tail 20
Get-Content logs/error.log -Tail 20

Full log reading is prohibited unless GPT-5.5 recovery mode needs a specific error range.

---

## 6. State Management

The agent must maintain:

reports/state.json
reports/progress_summary.md
reports/data_quality_report.md
reports/training_metrics_report.md
reports/recovery_log.md

If missing, create them.

state.json must include at least:

{
  "dataset": "Structured3D",
  "current_group": "panorama",
  "current_index": 0,
  "current_zip": "Structured3D_panorama_00.zip",
  "phase": "completed",
  "last_completed_zip": "Structured3D_panorama_00.zip",
  "next_zip": "Structured3D_panorama_01.zip",
  "valid_pairs": 3339,
  "skipped": 6,
  "cuda": true,
  "last_checkpoint": "checkpoints/structured3d/last.pt",
  "best_checkpoint": "checkpoints/structured3d/best.pt",
  "retry": 0,
  "last_error": null,
  "processed_zips": [
    "Structured3D_panorama_00.zip"
  ]
}

---

## 7. Dataset Order

Process all Structured3D archives in this exact order.

### Group 1: Panorama

Structured3D_panorama_00.zip
Structured3D_panorama_01.zip
Structured3D_panorama_02.zip
Structured3D_panorama_03.zip
Structured3D_panorama_04.zip
Structured3D_panorama_05.zip
Structured3D_panorama_06.zip
Structured3D_panorama_07.zip
Structured3D_panorama_08.zip
Structured3D_panorama_09.zip
Structured3D_panorama_10.zip
Structured3D_panorama_11.zip
Structured3D_panorama_12.zip
Structured3D_panorama_13.zip
Structured3D_panorama_14.zip
Structured3D_panorama_15.zip
Structured3D_panorama_16.zip
Structured3D_panorama_17.zip

### Group 2: Perspective Full

Structured3D_perspective_full_00.zip
Structured3D_perspective_full_01.zip
Structured3D_perspective_full_02.zip
Structured3D_perspective_full_03.zip
Structured3D_perspective_full_04.zip
Structured3D_perspective_full_05.zip
Structured3D_perspective_full_06.zip
Structured3D_perspective_full_07.zip
Structured3D_perspective_full_08.zip
Structured3D_perspective_full_09.zip
Structured3D_perspective_full_10.zip
Structured3D_perspective_full_11.zip
Structured3D_perspective_full_12.zip
Structured3D_perspective_full_13.zip
Structured3D_perspective_full_14.zip
Structured3D_perspective_full_15.zip
Structured3D_perspective_full_16.zip
Structured3D_perspective_full_17.zip

### Group 3: Perspective Empty

Structured3D_perspective_empty_00.zip
Structured3D_perspective_empty_01.zip
Structured3D_perspective_empty_02.zip
Structured3D_perspective_empty_03.zip
Structured3D_perspective_empty_04.zip
Structured3D_perspective_empty_05.zip
Structured3D_perspective_empty_06.zip
Structured3D_perspective_empty_07.zip
Structured3D_perspective_empty_08.zip
Structured3D_perspective_empty_09.zip
Structured3D_perspective_empty_10.zip
Structured3D_perspective_empty_11.zip
Structured3D_perspective_empty_12.zip
Structured3D_perspective_empty_13.zip
Structured3D_perspective_empty_14.zip
Structured3D_perspective_empty_15.zip
Structured3D_perspective_empty_16.zip
Structured3D_perspective_empty_17.zip

### Annotation

Structured3D_annotation_3d.zip must exist and be extracted once.

---

## 8. Download Rule

Do not guess URLs.

Base URL:

https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/

Download URL format:

https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/{zip_name}

Example:

Structured3D_panorama_01.zip

becomes:

https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/Structured3D_panorama_01.zip

Annotation URL:

https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/Structured3D_annotation_3d.zip

---

## 9. Full Auto Continuation Rule

Do not stop after one archive.

After each archive finishes training:

1. update reports/state.json
2. update reports/progress_summary.md
3. update reports/data_quality_report.md
4. update reports/training_metrics_report.md
5. commit and push allowed files
6. select the next zip from Dataset Order
7. continue automatically

Only stop when:

- all panorama 00-17 are processed
- all perspective_full 00-17 are processed
- all perspective_empty 00-17 are processed
- or Stop Conditions are reached

If state.json says panorama_00 is completed, continue with panorama_01.
If panorama_17 is completed, continue with perspective_full_00.
If perspective_full_17 is completed, continue with perspective_empty_00.
If perspective_empty_17 is completed, create final report and stop.

---

## 10. Automated Execution Loop Per Zip

For each zip:

1. determine current zip from reports/state.json
2. check zip existence under datasets/raw/structured3d
3. download if missing
4. wait for file size stabilization
5. run 7z test
6. allow partial archive errors
7. extract archive
8. skip unreadable images/masks
9. rebuild index
10. inspect masks
11. run GPU training with resume mode
12. validate best.pt and last.pt
13. update reports
14. git commit/push allowed files
15. move to next zip

---

## 11. Download Stability Check

Before testing or extracting any zip:

(Get-Item file).Length
Start-Sleep 5
(Get-Item file).Length

If size changes, wait.
Proceed only after size is unchanged for at least two consecutive checks.

---

## 12. Partial Error Handling

Allowed archive errors:

CRC Failed
Headers Error
Data Error

Rules:

- do not repeatedly re-download if local file size matches remote Content-Length
- use extractable files
- skip corrupted samples
- record archive errors in reports/data_quality_report.md
- proceed if valid pair count > 0

---

## 13. Index Rules

Index must include only readable samples.

Validation:

- image file exists
- mask file exists
- cv2.imread(image_path) succeeds
- cv2.imread(mask_path) succeeds

Required fields:

image_path
mask_path
scene_id
variant
source
mode

Recommended record:

{
  "id": "structured3d_scene_00000_485_simple",
  "source": "structured3d",
  "mode": "panorama",
  "variant": "simple",
  "scene_id": "scene_00000",
  "rendering_id": "485",
  "image_path": "...",
  "mask_path": "...",
  "width": 1024,
  "height": 512
}

---

## 14. Semantic Rules

Per-image random mapping is forbidden.

Required:

- maintain global deterministic mapping
- same color/value always maps to same class
- save mapping status to reports/semantic_mapping_report.md if implemented

Target classes:

0 background
1 wall
2 floor
3 ceiling
4 door
5 window
6 object
7 structure

---

## 15. Training Rules

CUDA is mandatory.

Required:

torch.cuda.is_available() == True

CPU fallback is forbidden.

Training must:

- use GPU
- use AMP when stable
- resume from last.pt if compatible
- ignore incompatible checkpoint only when class count changed
- maintain best.pt
- update last.pt
- output epoch summaries only
- report train loss, val loss, pixel accuracy, and mIoU

---

## 16. Logging Policy

Allowed output:

- archive status summary
- index summary
- epoch summary
- final metrics
- errors
- checkpoint saved

Forbidden output:

- full tqdm step logs
- full log file dumps
- repeated status spam

---

## 17. Metrics

Track:

train loss
validation loss
pixel accuracy
mIoU

Save to:

reports/training_metrics_report.md

---

## 18. Report Update Rules

After every archive, update all relevant reports.

### reports/state.json

Must update:

- current_group
- current_index
- current_zip
- phase
- last_completed_zip
- next_zip
- valid_pairs
- skipped
- cuda
- last_checkpoint
- best_checkpoint
- retry
- last_error
- processed_zips

### reports/progress_summary.md

Must include:

- current status
- last completed zip
- next zip
- total processed count
- current group
- last metrics summary
- next action

### reports/data_quality_report.md

Must include:

- archive name
- extraction status
- archive errors
- total candidates
- valid pairs
- skipped unreadable samples
- affected scenes if known

### reports/training_metrics_report.md

Must include:

- archive name
- final epoch
- train loss
- validation loss
- pixel accuracy
- mIoU
- checkpoint paths

### reports/recovery_log.md

Only append when:

- error occurs
- automatic recovery is applied
- GPT-5.5 recovery is needed

---

## 19. Error Handling

Automatic recovery:

- unreadable image or mask: skip and rebuild index
- stale index: rebuild
- partial archive corruption: extract usable files
- missing folder: create folder
- incompatible checkpoint: start fresh only for that model/class change

Escalate to GPT-5.5 recovery when:

- CUDA unavailable
- valid pairs = 0
- same error repeats twice
- training crash repeats twice
- dependency conflict
- git push fails repeatedly
- architecture decision is needed

---

## 20. Code Modification Rules

1. do not rewrite entire files unless necessary
2. make minimal targeted changes
3. preserve existing structure
4. avoid hardcoded absolute paths inside source code
5. prefer Path(__file__).resolve() for project-root discovery
6. separate preprocessing, dataset loading, training, metrics, and reporting
7. validate immediately after modification
8. if same error repeats twice, stop and request recovery supervisor
9. keep logs token-efficient
10. keep report files consistent with state.json

---

## 21. Git Automation

Before commit:

git status

Never commit forbidden files.

Allowed add command:

git add src requirements.txt README.md .gitignore reports harness.md

Commit:

git commit -m "auto: spatial fusion update"

Push:

git push

If no upstream:

git push -u origin main

If no remote:

stop and report. Do not invent a remote URL.

---

## 22. Stop Conditions

Stop only if:

- CUDA unavailable
- valid pairs = 0
- same error repeats twice
- disk is critically low
- forbidden files appear staged in git
- dependency install fails
- no git remote exists
- all target archives are completed

---

## 23. Final Goal

Complete:

- panorama 00-17
- perspective_full 00-17
- perspective_empty 00-17

Generate:

- final checkpoint
- final training metrics report
- final data quality report
- final progress summary
- clean git status

---

## 24. Final Output

[FINAL COMPLETION REPORT]

1. datasets processed:
2. valid samples:
3. skipped samples:
4. final train loss:
5. final val loss:
6. final pixel accuracy:
7. final mIoU:
8. best checkpoint:
9. last checkpoint:
10. reports updated:
11. git commit:
12. next step:

## 25. Autonomous Execution Policy

The agent must operate in fully autonomous mode.

User confirmation is NOT required for any of the following:

- file move (Move-Item)
- file creation
- directory creation
- dataset download
- archive extraction
- python execution
- pip install inside project venv
- index generation
- mask inspection
- training execution
- report updates
- git add / commit / push

The agent must NOT pause for confirmation.

If the environment requests confirmation, the agent must assume approval and proceed.

---

## 26. Safe Command Scope

The agent may execute without confirmation ONLY within:

C:\spatial-scan\ai-training

Never operate outside this directory.

---

## 27. Auto-Continue Guarantee

The agent must:

- never stop after a single dataset
- always continue to next dataset
- always resume from reports/state.json
- always continue after success OR partial failure

Stopping is ONLY allowed when Stop Conditions are met.

---

## 28. Git Auto Execution

The agent must automatically:

git add src requirements.txt README.md .gitignore reports harness.md
git commit -m "auto: spatial fusion update"
git push

No confirmation required.

---

## 29. Non-Blocking Execution Rule

If any command requests confirmation:

- assume YES
- proceed immediately

Do not wait for user input.

---

## 30. Silent Mode

The agent must:

- not request user decisions
- not pause execution
- not ask for confirmation

Only log actions internally.

User-facing output should be minimal.