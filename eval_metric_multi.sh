#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# GPUs
# ============================================================
GPUS=(2 3 4)

# ============================================================
# Songs: 5 in-distribution + 5 out-distribution
# ============================================================
SONGS=(
# In Distribution
  Adieu
  Beginners
  CanWeKissForever
  DoYouKnow
  Hello

# Out Distribution
  Alone
  EyesClosed
  OhneDich
  Paradise
  SomewhereOnlyWeKnow

# other's report / MIDI songs
# CMajorScaleTwoHands
# CMajorChordProgressionTwoHands
# TwinkleTwinkleRousseau
# DMajorScaleTwoHands
# NocturneRousseau
)

# ============================================================
# MIDI songs: add --use-midi only when song is in this list
# ============================================================
MIDI_SONGS=(
  TwinkleTwinkleLittleStar
  CMajorScaleOneHand
  CMajorScaleTwoHands
  DMajorScaleOneHand
  DMajorScaleTwoHands
  CMajorChordProgressionTwoHands
  TwinkleTwinkleRousseau
  NocturneRousseau
)

is_midi_song() {
  local song="$1"
  local midi_song
  for midi_song in "${MIDI_SONGS[@]}"; do
    if [[ "${song}" == "${midi_song}" ]]; then
      return 0
    fi
  done
  return 1
}

# ============================================================
# Checkpoints
# ============================================================
# Author-provided DDPM checkpoint.
GIVEN_AE_CKPT="given_ckpt/checkpoint_ae.ckpt"
GIVEN_DDPM_HL_CKPT="given_ckpt/checkpoint_high_level.ckpt"
GIVEN_DDPM_LL_CKPT="given_ckpt/checkpoint_low_level.ckpt"

# Reproduced DDPM checkpoint.
REPRODUCED_AE_CKPT="reproduced_ckpt/checkpoint_ae.ckpt"
REPRODUCED_DDPM_HL_CKPT="reproduced_ckpt/dataset_hl_without_fingering.ckpt"
REPRODUCED_DDPM_LL_CKPT="reproduced_ckpt/dataset_ll.ckpt"

# Flow checkpoint. Keep the same AE as the reproduced run unless you intentionally change it.
FLOW_AE_CKPT="${REPRODUCED_AE_CKPT}"
FLOW_HL_CKPT="flow/ckpts/checkpoint_FM-HL-dataset_hl_without_fingering.ckpt"
FLOW_LL_CKPT="flow/ckpts/checkpoint_FM-LL-dataset_ll.ckpt"

# ============================================================
# Logs
# ============================================================
LOGDIR="logs/eval_metrics_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOGDIR}"

FAILED_FILE="${LOGDIR}/failed_tasks.txt"
touch "${FAILED_FILE}"

# ============================================================
# Build task list
# Format:
# song|policy|label|flow_steps
# ============================================================
TASKS=()

for song in "${SONGS[@]}"; do
  # DDPM using author-provided checkpoints.
  TASKS+=("${song}|ddpm_given|ddpm_given|")

  # DDPM using reproduced checkpoints.
  TASKS+=("${song}|ddpm|ddpm|")

  # Flow matching variants.
  TASKS+=("${song}|flow|fm10|10")
  TASKS+=("${song}|flow|fm20|20")
  TASKS+=("${song}|flow|fm50|50")
done

TOTAL=${#TASKS[@]}

# ============================================================
# Shared task index
# ============================================================
INDEX_FILE="${LOGDIR}/task_index.txt"
LOCK_FILE="${LOGDIR}/task.lock"
echo 0 > "${INDEX_FILE}"

next_task() {
  {
    flock 200

    local idx
    idx=$(cat "${INDEX_FILE}")

    if (( idx >= TOTAL )); then
      echo "__DONE__"
    else
      echo $((idx + 1)) > "${INDEX_FILE}"
      echo "${TASKS[$idx]}"
    fi
  } 200>"${LOCK_FILE}"
}

# ============================================================
# Run one task
# ============================================================
run_task() {
  local gpu="$1"
  local song="$2"
  local policy="$3"
  local label="$4"
  local flow_steps="$5"

  local midi_args=()
  if is_midi_song "${song}"; then
    midi_args+=(--use-midi)
  fi

  if [[ "${policy}" == "ddpm_given" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python pianomime/eval_metrics.py "${song}" \
      --policy ddpm \
      --ae-ckpt "${GIVEN_AE_CKPT}" \
      --high-level-ckpt "${GIVEN_DDPM_HL_CKPT}" \
      --low-level-ckpt "${GIVEN_DDPM_LL_CKPT}" \
      --label "${label}" \
      "${midi_args[@]}"

  elif [[ "${policy}" == "ddpm" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python pianomime/eval_metrics.py "${song}" \
      --policy ddpm \
      --ae-ckpt "${REPRODUCED_AE_CKPT}" \
      --high-level-ckpt "${REPRODUCED_DDPM_HL_CKPT}" \
      --low-level-ckpt "${REPRODUCED_DDPM_LL_CKPT}" \
      --label "${label}" \
      "${midi_args[@]}"

  elif [[ "${policy}" == "flow" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python pianomime/eval_metrics.py "${song}" \
      --policy flow \
      --ae-ckpt "${FLOW_AE_CKPT}" \
      --high-level-ckpt "${FLOW_HL_CKPT}" \
      --low-level-ckpt "${FLOW_LL_CKPT}" \
      --flow-steps "${flow_steps}" \
      --flow-solver euler \
      --flow-clip-mode final \
      --label "${label}" \
      "${midi_args[@]}"

  else
    echo "Unknown policy: ${policy}"
    return 1
  fi
}

# ============================================================
# Worker: one worker per GPU
# ============================================================
worker() {
  local gpu="$1"

  while true; do
    local item
    item=$(next_task)

    if [[ "${item}" == "__DONE__" ]]; then
      echo "[GPU ${gpu}] no more tasks."
      break
    fi

    local song policy label flow_steps
    IFS='|' read -r song policy label flow_steps <<< "${item}"

    local log_file="${LOGDIR}/${song}_${label}_gpu${gpu}.log"

    echo "[$(date '+%F %T')] [GPU ${gpu}] START ${song} ${label}" | tee -a "${log_file}"

    if run_task "${gpu}" "${song}" "${policy}" "${label}" "${flow_steps}" >> "${log_file}" 2>&1; then
      echo "[$(date '+%F %T')] [GPU ${gpu}] DONE  ${song} ${label}" | tee -a "${log_file}"
    else
      echo "[$(date '+%F %T')] [GPU ${gpu}] FAIL  ${song} ${label}" | tee -a "${log_file}"
      echo "${song} ${label} gpu=${gpu}" >> "${FAILED_FILE}"
    fi
  done
}

# ============================================================
# Launch workers
# ============================================================
echo "Total tasks: ${TOTAL}"
echo "GPUs: ${GPUS[*]}"
echo "Logs: ${LOGDIR}"

echo "DDPM given ckpt:      ${GIVEN_DDPM_HL_CKPT} / ${GIVEN_DDPM_LL_CKPT}"
echo "DDPM reproduced ckpt: ${REPRODUCED_DDPM_HL_CKPT} / ${REPRODUCED_DDPM_LL_CKPT}"
echo "Flow ckpt:            ${FLOW_HL_CKPT} / ${FLOW_LL_CKPT}"

for gpu in "${GPUS[@]}"; do
  worker "${gpu}" &
done

wait

echo "All tasks finished."
echo "Logs saved to: ${LOGDIR}"

if [[ -s "${FAILED_FILE}" ]]; then
  echo "Some tasks failed. See: ${FAILED_FILE}"
else
  echo "No failed tasks."
fi
