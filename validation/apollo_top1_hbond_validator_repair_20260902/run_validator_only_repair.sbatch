#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=salsburygrp
#SBATCH --partition=small
#SBATCH --qos=normal
#SBATCH --job-name=top1-hbond-gate-v2
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/deac/phy/salsburyGrp/TOP1_EDU_1A36_2026-07-16/source/simulation_protocol_208ns_20260715/analysis/top1_edu_hbond_validator_repair_20260902_attempt02/logs/%j-validator.out
#SBATCH --error=/deac/phy/salsburyGrp/TOP1_EDU_1A36_2026-07-16/source/simulation_protocol_208ns_20260715/analysis/top1_edu_hbond_validator_repair_20260902_attempt02/logs/%j-validator.err
set -euo pipefail
umask 0002

BASE=/deac/phy/salsburyGrp/TOP1_EDU_1A36_2026-07-16/source/simulation_protocol_208ns_20260715/analysis
FAILED_ROOT="$BASE/top1_edu_analysis_replacements_20260902_attempt01"
REPAIR_ROOT="$BASE/top1_edu_hbond_validator_repair_20260902_attempt02"
SOURCE="$BASE/salsbury_md_analysis_main_3641081_hbond_calibration_20260902/source"
PYTHON=/deac/phy/salsburyGrp/software/salsbury-md-analysis/environments/v76/bin/python3.12
VALIDATOR="$REPAIR_ROOT/validate_completed_hbond_reports.py"
VALIDATOR_SHA=292a7eae5dd9f606145f394d1fc48c2c5404e32e49a5655798601a9d0c1428be
FRAME_SAMPLING_SHA=da14c4afd023fe3f9dc1ab859940b81196c15b910f07939c4522d6bb0bfc3dc8

D_REPORT="$FAILED_ROOT/replacements/D_hbonds_attempt01/report.json.tmp.8298182"
D_SUMMARY="$D_REPORT.summary.json"
T_REPORT="$FAILED_ROOT/replacements/T_hbonds_attempt01/report.json.tmp.8298183"
T_SUMMARY="$T_REPORT.summary.json"

D_PREP="$BASE/salsbury_md_analysis_main_11a9707_replacement_20260902/D/D_770g_1node_28cpu_2x_120h_replacement_20260902/prepared"
T_PREP="$BASE/salsbury_md_analysis_main_11a9707_replacement_20260902/T/T_770g_2node_24cpu_2x_135h_replacement_20260902/prepared"

mkdir -p "$REPAIR_ROOT/logs" "$REPAIR_ROOT/replacements/D_hbonds" "$REPAIR_ROOT/replacements/T_hbonds"
export PYTHONPATH="$SOURCE/src"
export PYTHONPYCACHEPREFIX="$REPAIR_ROOT/python-cache"

test "$(sha256sum "$VALIDATOR" | cut -d' ' -f1)" = "$VALIDATOR_SHA"
test "$(sha256sum "$SOURCE/src/salsbury_md_analysis/frame_sampling.py" | cut -d' ' -f1)" = "$FRAME_SAMPLING_SHA"

"$PYTHON" "$VALIDATOR" \
  --report "$D_REPORT" \
  --summary "$D_SUMMARY" \
  --expected-report-sha256 52523fcf0ff5bad8bcad33c4dea2f1f6e4d795e529d52bed7e239c8a88bb92ef \
  --expected-summary-sha256 3dea0adb65fb9730b3b6173e1e7d11af2898678af94d69c8bd572006d2026410 \
  --expected-project-sha256 926d43941da07661b5eade01b74738f037e0a6ae0ea3fd0afda34a4147398054 \
  --expected-stride 15 \
  --expected-systems D0,D1,T0-DNA_CONTEXT,T1-DNA_CONTEXT \
  --install-report "$REPAIR_ROOT/replacements/D_hbonds/report.json" \
  --install-summary "$REPAIR_ROOT/replacements/D_hbonds/report.json.summary.json" \
  --install-report "$D_PREP/results/hydrogen-bond-discovery/report.json" \
  --install-summary "$D_PREP/results/hydrogen-bond-discovery/report.json.summary.json" \
  > "$REPAIR_ROOT/replacements/D_hbonds/VALIDATION_AND_INSTALLATION_GATE.json"

"$PYTHON" "$VALIDATOR" \
  --report "$T_REPORT" \
  --summary "$T_SUMMARY" \
  --expected-report-sha256 64b3be98fbc00874aa295fba0a35e3fa048f54dc9d10225ca3b22e3bb538ecd3 \
  --expected-summary-sha256 76d3d8cc0a04e9f99d300d6f6ef5c6f04c2adb16f15528c5ba686602246f2427 \
  --expected-project-sha256 f97d9829588646cedb7b64033104b2f699270ed97bc299b5ba0aaf1d9419d7ff \
  --expected-stride 29 \
  --expected-systems T0,T1 \
  --install-report "$REPAIR_ROOT/replacements/T_hbonds/report.json" \
  --install-summary "$REPAIR_ROOT/replacements/T_hbonds/report.json.summary.json" \
  --install-report "$T_PREP/results/hydrogen-bond-discovery/report.json" \
  --install-summary "$T_PREP/results/hydrogen-bond-discovery/report.json.summary.json" \
  > "$REPAIR_ROOT/replacements/T_hbonds/VALIDATION_AND_INSTALLATION_GATE.json"

sha256sum \
  "$VALIDATOR" \
  "$REPAIR_ROOT/replacements/D_hbonds/report.json" \
  "$REPAIR_ROOT/replacements/D_hbonds/report.json.summary.json" \
  "$REPAIR_ROOT/replacements/D_hbonds/VALIDATION_AND_INSTALLATION_GATE.json" \
  "$REPAIR_ROOT/replacements/T_hbonds/report.json" \
  "$REPAIR_ROOT/replacements/T_hbonds/report.json.summary.json" \
  "$REPAIR_ROOT/replacements/T_hbonds/VALIDATION_AND_INSTALLATION_GATE.json" \
  > "$REPAIR_ROOT/OUTPUT_SHA256SUMS.txt"
