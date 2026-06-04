#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# Global switches
# ============================================================
# ENABLE_IK=1: pass --enable-ik to every eval_metrics.py call.
# ENABLE_IK=0: pass --no-enable-ik to every eval_metrics.py call.
ENABLE_IK=${ENABLE_IK:-1}

# ============================================================
# GPUs
# ============================================================
GPUS=(2 3 4)

# ============================================================
# Songs: edit this list directly
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

# MIDI songs
  TwinkleTwinkleLittleStar
  CMajorScaleOneHand
  CMajorScaleTwoHands
  DMajorScaleOneHand
  DMajorScaleTwoHands
  CMajorChordProgressionTwoHands
  TwinkleTwinkleRousseau
  NocturneRousseau
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
# Sampler step sweeps
# ============================================================
DDIM_STEPS=(25 50)

# Flow Matching is evaluated in two protocols:
#   euler: speed-oriented protocol
#   heun:  quality-oriented protocol, about 2x network evaluations per step
FLOW_EULER_STEPS=(10 20 50)
FLOW_HEUN_STEPS=(10 20 50)
FLOW_EULER_CLIP_MODE=${FLOW_EULER_CLIP_MODE:-final}
FLOW_HEUN_CLIP_MODE=${FLOW_HEUN_CLIP_MODE:-none}

# ============================================================
# Datasets / normalization stats
# These zarr paths are read by eval scripts for normalization statistics.
# Override from shell if given/reproduced checkpoints use different stats.
# Example:
#   GIVEN_DATASET_HL=official_dataset_hl.zarr GIVEN_DATASET_LL=official_dataset_ll.zarr bash pianomime/eval_metric_multi.sh
# ============================================================
GIVEN_DATASET_HL=${GIVEN_DATASET_HL:-dataset_hl.zarr}
GIVEN_DATASET_LL=${GIVEN_DATASET_LL:-dataset_ll.zarr}

REPRODUCED_DATASET_HL=${REPRODUCED_DATASET_HL:-dataset_hl.zarr}
REPRODUCED_DATASET_LL=${REPRODUCED_DATASET_LL:-dataset_ll.zarr}

FLOW_DATASET_HL=${FLOW_DATASET_HL:-${REPRODUCED_DATASET_HL}}
FLOW_DATASET_LL=${FLOW_DATASET_LL:-${REPRODUCED_DATASET_LL}}

# ============================================================
# Checkpoints
# ============================================================
# Author-provided DDPM / DDIM checkpoint.
GIVEN_AE_CKPT="given_ckpt/checkpoint_ae.ckpt"
GIVEN_DDPM_HL_CKPT="given_ckpt/checkpoint_high_level.ckpt"
GIVEN_DDPM_LL_CKPT="given_ckpt/checkpoint_low_level.ckpt"

# Reproduced DDPM / DDIM checkpoint.
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
METRICS_LOGDIR="${LOGDIR}/metrics"
mkdir -p "${LOGDIR}" "${METRICS_LOGDIR}"

FAILED_FILE="${LOGDIR}/failed_tasks.txt"
touch "${FAILED_FILE}"

# ============================================================
# Build task list
# Format:
#   song|policy|label|steps|solver|clip_mode
#
# policy values:
#   ddpm_given, ddpm, ddim_given, ddim, flow
#
# Scheduling rule:
#   We build the queue by experiment first and song second. The scheduler also
#   prevents two tasks for the same song from running at the same time, because
#   all eval scripts write/read pianomime/multi_task/trajectories/<song>_*.npy.
# ============================================================
TASKS=()

add_task_for_all_songs() {
  local policy="$1"
  local label="$2"
  local steps="$3"
  local solver="$4"
  local clip_mode="$5"

  local song
  for song in "${SONGS[@]}"; do
    TASKS+=("${song}|${policy}|${label}|${steps}|${solver}|${clip_mode}")
  done
}

# DDPM.
add_task_for_all_songs "ddpm_given" "ddpm_given" "" "" ""
add_task_for_all_songs "ddpm" "ddpm" "" "" ""

# DDIM.
for step in "${DDIM_STEPS[@]}"; do
  add_task_for_all_songs "ddim_given" "ddim_given${step}" "${step}" "" ""
done
for step in "${DDIM_STEPS[@]}"; do
  add_task_for_all_songs "ddim" "ddim${step}" "${step}" "" ""
done

# Flow Matching, speed-oriented Euler protocol.
for step in "${FLOW_EULER_STEPS[@]}"; do
  add_task_for_all_songs "flow" "fm_euler${step}" "${step}" "euler" "${FLOW_EULER_CLIP_MODE}"
done

# Flow Matching, quality-oriented Heun protocol.
for step in "${FLOW_HEUN_STEPS[@]}"; do
  add_task_for_all_songs "flow" "fm_heun${step}" "${step}" "heun" "${FLOW_HEUN_CLIP_MODE}"
done

TOTAL=${#TASKS[@]}

# ============================================================
# Shared task scheduler
# ============================================================
STATUS_FILE="${LOGDIR}/task_status.txt"
ACTIVE_FILE="${LOGDIR}/active_songs.txt"
LOCK_FILE="${LOGDIR}/task.lock"
: > "${STATUS_FILE}"
: > "${ACTIVE_FILE}"
for ((i=0; i<TOTAL; i++)); do
  echo "0" >> "${STATUS_FILE}"   # 0=pending, 1=running, 2=done
done

get_status_line() {
  local line_no="$1"
  sed -n "${line_no}p" "${STATUS_FILE}"
}

set_status_line() {
  local line_no="$1"
  local value="$2"
  awk -v n="${line_no}" -v v="${value}" 'NR==n {$0=v} {print}' "${STATUS_FILE}" > "${STATUS_FILE}.tmp"
  mv "${STATUS_FILE}.tmp" "${STATUS_FILE}"
}

song_is_active() {
  local song="$1"
  grep -Fxq "${song}" "${ACTIVE_FILE}"
}

activate_song() {
  local song="$1"
  echo "${song}" >> "${ACTIVE_FILE}"
}

release_song() {
  local song="$1"
  grep -Fxv "${song}" "${ACTIVE_FILE}" > "${ACTIVE_FILE}.tmp" || true
  mv "${ACTIVE_FILE}.tmp" "${ACTIVE_FILE}"
}

next_task() {
  {
    flock 200

    local pending=0
    local running=0
    local i line_no status item song policy label steps solver clip_mode

    for ((i=0; i<TOTAL; i++)); do
      line_no=$((i + 1))
      status=$(get_status_line "${line_no}")

      if [[ "${status}" == "0" ]]; then
        pending=1
        item="${TASKS[$i]}"
        IFS='|' read -r song policy label steps solver clip_mode <<< "${item}"

        # Do not dispatch a second task for the same song while one is running.
        if ! song_is_active "${song}"; then
          set_status_line "${line_no}" "1"
          activate_song "${song}"
          echo "${i}|${item}"
          exit 0
        fi
      elif [[ "${status}" == "1" ]]; then
        running=1
      fi
    done

    if (( pending == 0 && running == 0 )); then
      echo "__DONE__"
    else
      echo "__WAIT__"
    fi
  } 200>"${LOCK_FILE}"
}

finish_task() {
  local task_idx="$1"
  local song="$2"
  {
    flock 200
    set_status_line $((task_idx + 1)) "2"
    release_song "${song}"
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
  local steps="$5"
  local solver="$6"
  local clip_mode="$7"

  local midi_args=()
  if is_midi_song "${song}"; then
    midi_args+=(--use-midi)
  fi

  local ik_args=()
  if [[ "${ENABLE_IK}" == "1" ]]; then
    ik_args+=(--enable-ik)
  else
    ik_args+=(--no-enable-ik)
  fi

  local common_args=(--log-dir "${METRICS_LOGDIR}")

  if [[ "${policy}" == "ddpm_given" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python pianomime/eval_metrics.py "${song}" \
      --policy ddpm \
      --ae-ckpt "${GIVEN_AE_CKPT}" \
      --high-level-ckpt "${GIVEN_DDPM_HL_CKPT}" \
      --low-level-ckpt "${GIVEN_DDPM_LL_CKPT}" \
      --dataset-hl "${GIVEN_DATASET_HL}" \
      --dataset-ll "${GIVEN_DATASET_LL}" \
      --label "${label}" \
      "${common_args[@]}" \
      "${midi_args[@]}" \
      "${ik_args[@]}"

  elif [[ "${policy}" == "ddpm" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python pianomime/eval_metrics.py "${song}" \
      --policy ddpm \
      --ae-ckpt "${REPRODUCED_AE_CKPT}" \
      --high-level-ckpt "${REPRODUCED_DDPM_HL_CKPT}" \
      --low-level-ckpt "${REPRODUCED_DDPM_LL_CKPT}" \
      --dataset-hl "${REPRODUCED_DATASET_HL}" \
      --dataset-ll "${REPRODUCED_DATASET_LL}" \
      --label "${label}" \
      "${common_args[@]}" \
      "${midi_args[@]}" \
      "${ik_args[@]}"

  elif [[ "${policy}" == "ddim_given" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python pianomime/eval_metrics.py "${song}" \
      --policy ddim \
      --ae-ckpt "${GIVEN_AE_CKPT}" \
      --high-level-ckpt "${GIVEN_DDPM_HL_CKPT}" \
      --low-level-ckpt "${GIVEN_DDPM_LL_CKPT}" \
      --dataset-hl "${GIVEN_DATASET_HL}" \
      --dataset-ll "${GIVEN_DATASET_LL}" \
      --ddim-steps "${steps}" \
      --label "${label}" \
      "${common_args[@]}" \
      "${midi_args[@]}" \
      "${ik_args[@]}"

  elif [[ "${policy}" == "ddim" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python pianomime/eval_metrics.py "${song}" \
      --policy ddim \
      --ae-ckpt "${REPRODUCED_AE_CKPT}" \
      --high-level-ckpt "${REPRODUCED_DDPM_HL_CKPT}" \
      --low-level-ckpt "${REPRODUCED_DDPM_LL_CKPT}" \
      --dataset-hl "${REPRODUCED_DATASET_HL}" \
      --dataset-ll "${REPRODUCED_DATASET_LL}" \
      --ddim-steps "${steps}" \
      --label "${label}" \
      "${common_args[@]}" \
      "${midi_args[@]}" \
      "${ik_args[@]}"

  elif [[ "${policy}" == "flow" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python pianomime/eval_metrics.py "${song}" \
      --policy flow \
      --ae-ckpt "${FLOW_AE_CKPT}" \
      --high-level-ckpt "${FLOW_HL_CKPT}" \
      --low-level-ckpt "${FLOW_LL_CKPT}" \
      --dataset-hl "${FLOW_DATASET_HL}" \
      --dataset-ll "${FLOW_DATASET_LL}" \
      --flow-steps "${steps}" \
      --flow-solver "${solver}" \
      --flow-clip-mode "${clip_mode}" \
      --flow-hl-steps "${steps}" \
      --flow-ll-steps "${steps}" \
      --flow-hl-solver "${solver}" \
      --flow-ll-solver "${solver}" \
      --flow-hl-clip-mode "${clip_mode}" \
      --flow-ll-clip-mode "${clip_mode}" \
      --label "${label}" \
      "${common_args[@]}" \
      "${midi_args[@]}" \
      "${ik_args[@]}"

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

    if [[ "${item}" == "__WAIT__" ]]; then
      # All remaining pending tasks conflict with currently active songs.
      # Sleep briefly and try again, instead of blocking a GPU inside eval_metrics.py.
      sleep 5
      continue
    fi

    local task_idx song policy label steps solver clip_mode
    IFS='|' read -r task_idx song policy label steps solver clip_mode <<< "${item}"

    local log_file="${LOGDIR}/${song}_${label}_gpu${gpu}.log"

    echo "[$(date '+%F %T')] [GPU ${gpu}] START ${song} ${label} enable_ik=${ENABLE_IK}" | tee -a "${log_file}"

    if run_task "${gpu}" "${song}" "${policy}" "${label}" "${steps}" "${solver}" "${clip_mode}" >> "${log_file}" 2>&1; then
      echo "[$(date '+%F %T')] [GPU ${gpu}] DONE  ${song} ${label}" | tee -a "${log_file}"
    else
      echo "[$(date '+%F %T')] [GPU ${gpu}] FAIL  ${song} ${label}" | tee -a "${log_file}"
      echo "${song} ${label} gpu=${gpu}" >> "${FAILED_FILE}"
    fi

    finish_task "${task_idx}" "${song}"
  done
}

# ============================================================
# Launch workers
# ============================================================
echo "Total tasks: ${TOTAL}"
echo "GPUs: ${GPUS[*]}"
echo "Logs: ${LOGDIR}"
echo "Metrics CSV: ${METRICS_LOGDIR}/results.csv"
echo "ENABLE_IK: ${ENABLE_IK}"
echo "DDIM steps: ${DDIM_STEPS[*]}"
echo "Flow Euler steps: ${FLOW_EULER_STEPS[*]} clip=${FLOW_EULER_CLIP_MODE}"
echo "Flow Heun steps:  ${FLOW_HEUN_STEPS[*]} clip=${FLOW_HEUN_CLIP_MODE}"
echo "DDPM/DDIM given ckpt:      ${GIVEN_DDPM_HL_CKPT} / ${GIVEN_DDPM_LL_CKPT}"
echo "DDPM/DDIM reproduced ckpt: ${REPRODUCED_DDPM_HL_CKPT} / ${REPRODUCED_DDPM_LL_CKPT}"
echo "Flow ckpt:                 ${FLOW_HL_CKPT} / ${FLOW_LL_CKPT}"
echo "Given dataset stats:       ${GIVEN_DATASET_HL} / ${GIVEN_DATASET_LL}"
echo "Reproduced dataset stats:  ${REPRODUCED_DATASET_HL} / ${REPRODUCED_DATASET_LL}"
echo "Flow dataset stats:        ${FLOW_DATASET_HL} / ${FLOW_DATASET_LL}"

for gpu in "${GPUS[@]}"; do
  worker "${gpu}" &
done

wait

echo "All tasks finished."
echo "Logs saved to: ${LOGDIR}"
echo "Metrics CSV: ${METRICS_LOGDIR}/results.csv"

if [[ -s "${FAILED_FILE}" ]]; then
  echo "Some tasks failed. See: ${FAILED_FILE}"
else
  echo "No failed tasks."
fi
