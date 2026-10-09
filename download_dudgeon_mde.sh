#!/usr/bin/env bash

set -u
set -o pipefail

DEFAULT_DEST="/Volumes/HP P900/Dudgeon_2013_MDE"
DEST_ROOT="$DEFAULT_DEST"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  download_dudgeon_mde.sh [--dry-run] [destination]

Examples:
  download_dudgeon_mde.sh --dry-run
  download_dudgeon_mde.sh
  download_dudgeon_mde.sh "/Volumes/HP P900/Dudgeon_2013_MDE"

The script downloads the recommended Dudgeon 2013 MDE packages sequentially.
Partial downloads use a .part suffix and are resumed automatically on the next run.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      DEST_ROOT="$1"
      shift
      if [[ $# -gt 0 ]]; then
        printf 'Only one destination may be supplied.\n' >&2
        exit 2
      fi
      ;;
  esac
done

VOLUME_ROOT="/Volumes/HP P900"
MANIFEST_DIR="$DEST_ROOT/00_manifest"
ARCHIVE_DIR="$DEST_ROOT/01_source_zips"
REPORT_DIR="$DEST_ROOT/02_reports"
MBES_DIR="$DEST_ROOT/10_mbes"
SSS_DIR="$DEST_ROOT/20_sss"
MAG_DIR="$DEST_ROOT/30_magnetometer"
GIS_DIR="$DEST_ROOT/40_gis"

# filename|expected bytes|official MDE download URL|category
DOWNLOADS=(
  "01_SSS_Contact_Lists.zip|2641093|https://www.marinedataexchange.co.uk/pub/TCE/23-316-Side-scan%20Sonar%20Contact%20Lists.zip|SSS contacts"
  "02_Side_Scan_Sonar_Data.zip|395051694|https://www.marinedataexchange.co.uk/pub/TCE/23-316-Side-Scan%20Sonar%20Data.zip|SSS images"
  "03_GIS_SSDM.zip|29903962|https://www.marinedataexchange.co.uk/pub/TCE/23-316-GIS(SSDM).zip|GIS SSDM"
  "04_Magnetometer_Data_and_Contact_Lists.zip|3824332953|https://www.marinedataexchange.co.uk/pub/TCE/23_316_Magnetometer%20Data%20and%20Contact%20Lists.zip|Magnetometer"
  "05_MBES_DTM_ASCII.zip|7690298159|https://www.marinedataexchange.co.uk/pub/TCE/23-316-DTM(ASCII).zip|MBES DTM"
  "06_GIS_Files.zip|4387749842|https://www.marinedataexchange.co.uk/pub/TCE/23-316-GIS%20Files.zip|GIS rasters and vectors"
  "07_Field_Report.zip|215107346|https://www.marinedataexchange.co.uk/pub/TCE/23_316_Field%20Report.zip|Field report"
  "08_Interpretive_Report.zip|25154871|https://www.marinedataexchange.co.uk/pub/TCE/23_316_Interpretive%20Report.zip|Interpretive report"
)

EXPECTED_TOTAL=16570239920
CURRENT_NAME=""
CURRENT_PART=""

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

human_bytes() {
  awk -v value="$1" 'BEGIN {
    split("B KB MB GB TB", units, " ");
    unit_index = 1;
    while (value >= 1000 && unit_index < 5) { value /= 1000; unit_index++ }
    printf "%.2f %s", value, units[unit_index]
  }'
}

file_size() {
  local target_file="$1"
  if stat -f '%z' "$target_file" >/dev/null 2>&1; then
    stat -f '%z' "$target_file"
  else
    stat -c '%s' "$target_file"
  fi
}

log_line() {
  local message="[$(timestamp)] $*"
  printf '%s\n' "$message"
  if [[ -n "${LOG_FILE:-}" ]]; then
    printf '%s\n' "$message" >> "$LOG_FILE"
  fi
}

on_interrupt() {
  printf '\n'
  log_line "Download interrupted. Partial data has been kept."
  if [[ -n "$CURRENT_NAME" ]]; then
    log_line "Current file: $CURRENT_NAME"
    log_line "Partial file: $CURRENT_PART"
  fi
  log_line "Run the same command again to resume."
  exit 130
}

trap on_interrupt INT TERM HUP

if [[ ! -d "$VOLUME_ROOT" ]]; then
  printf 'External drive is not mounted at: %s\n' "$VOLUME_ROOT" >&2
  exit 1
fi

if [[ ! -w "$VOLUME_ROOT" ]]; then
  printf 'External drive is not writable: %s\n' "$VOLUME_ROOT" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  printf 'curl is required but was not found.\n' >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  printf 'unzip is required but was not found.\n' >&2
  exit 1
fi

if ! command -v shasum >/dev/null 2>&1; then
  printf 'shasum is required but was not found.\n' >&2
  exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Destination: %s\n' "$DEST_ROOT"
  printf 'Expected download: %s (%s bytes)\n\n' "$(human_bytes "$EXPECTED_TOTAL")" "$EXPECTED_TOTAL"
  printf '%-46s %12s  %s\n' 'FILE' 'SIZE' 'CATEGORY'
  printf '%-46s %12s  %s\n' '----' '----' '--------'
  for entry in "${DOWNLOADS[@]}"; do
    IFS='|' read -r name expected url category <<< "$entry"
    printf '%-46s %12s  %s\n' "$name" "$(human_bytes "$expected")" "$category"
  done
  exit 0
fi

# Keep the Mac awake for this foreground, user-started download session.
if command -v caffeinate >/dev/null 2>&1 && [[ "${DUDGEON_CAFFEINATED:-0}" != "1" ]]; then
  exec env DUDGEON_CAFFEINATED=1 caffeinate -dimsu "$0" "$DEST_ROOT"
fi

mkdir -p \
  "$MANIFEST_DIR" \
  "$ARCHIVE_DIR" \
  "$REPORT_DIR" \
  "$MBES_DIR" \
  "$SSS_DIR" \
  "$MAG_DIR" \
  "$GIS_DIR"

LOG_FILE="$MANIFEST_DIR/download.log"
MANIFEST_FILE="$MANIFEST_DIR/mde_download_manifest.tsv"
CHECKSUM_FILE="$MANIFEST_DIR/SHA256SUMS.txt"

printf 'filename\texpected_bytes\tcategory\turl\n' > "$MANIFEST_FILE"
for entry in "${DOWNLOADS[@]}"; do
  IFS='|' read -r name expected url category <<< "$entry"
  printf '%s\t%s\t%s\t%s\n' "$name" "$expected" "$category" "$url" >> "$MANIFEST_FILE"
done

remaining_bytes=0
for entry in "${DOWNLOADS[@]}"; do
  IFS='|' read -r name expected url category <<< "$entry"
  final_file="$ARCHIVE_DIR/$name"
  part_file="$final_file.part"

  if [[ -f "$final_file" ]]; then
    actual=$(file_size "$final_file")
    if [[ "$actual" -ne "$expected" ]]; then
      printf 'Existing completed file has the wrong size: %s\n' "$final_file" >&2
      printf 'Expected %s bytes, found %s bytes. Nothing was overwritten.\n' "$expected" "$actual" >&2
      exit 1
    fi
    continue
  fi

  partial_size=0
  if [[ -f "$part_file" ]]; then
    partial_size=$(file_size "$part_file")
    if [[ "$partial_size" -gt "$expected" ]]; then
      printf 'Partial file is larger than expected: %s\n' "$part_file" >&2
      printf 'Nothing was overwritten. Move it aside and run the script again.\n' >&2
      exit 1
    fi
  fi
  remaining_bytes=$((remaining_bytes + expected - partial_size))
done

free_kib=$(df -Pk "$VOLUME_ROOT" | awk 'NR == 2 {print $4}')
free_bytes=$((free_kib * 1024))
safety_buffer=$((2 * 1024 * 1024 * 1024))
if [[ "$free_bytes" -lt $((remaining_bytes + safety_buffer)) ]]; then
  printf 'Not enough free space on %s.\n' "$VOLUME_ROOT" >&2
  printf 'Remaining download: %s; free: %s; required safety buffer: 2 GiB.\n' \
    "$(human_bytes "$remaining_bytes")" "$(human_bytes "$free_bytes")" >&2
  exit 1
fi

log_line "Starting Dudgeon 2013 MDE download."
log_line "Destination: $DEST_ROOT"
log_line "Remaining download: $(human_bytes "$remaining_bytes")"
log_line "Free space: $(human_bytes "$free_bytes")"

completed_names=()

for entry in "${DOWNLOADS[@]}"; do
  IFS='|' read -r name expected url category <<< "$entry"
  CURRENT_NAME="$name"
  final_file="$ARCHIVE_DIR/$name"
  part_file="$final_file.part"
  CURRENT_PART="$part_file"

  if [[ -f "$final_file" ]]; then
    log_line "Already complete; skipping $name"
    completed_names+=("$name")
    continue
  fi

  if [[ -f "$part_file" ]]; then
    partial_size=$(file_size "$part_file")
    log_line "Resuming $name at $(human_bytes "$partial_size") of $(human_bytes "$expected")."
  else
    log_line "Downloading $name ($(human_bytes "$expected"), $category)."
  fi

  curl \
    --fail \
    --location \
    --continue-at - \
    --retry 30 \
    --retry-delay 5 \
    --retry-all-errors \
    --connect-timeout 30 \
    --speed-limit 1024 \
    --speed-time 180 \
    --progress-bar \
    --output "$part_file" \
    "$url"
  curl_status=$?

  if [[ "$curl_status" -ne 0 ]]; then
    log_line "curl stopped with status $curl_status. The partial file was kept for resuming."
    exit "$curl_status"
  fi

  actual=$(file_size "$part_file")
  if [[ "$actual" -ne "$expected" ]]; then
    log_line "Size verification failed for $name: expected $expected bytes, got $actual bytes."
    log_line "The .part file was kept; nothing was overwritten."
    exit 1
  fi

  log_line "Testing ZIP integrity for $name."
  if ! unzip -tq "$part_file" >/dev/null; then
    log_line "ZIP integrity test failed for $name. The .part file was kept."
    exit 1
  fi

  mv "$part_file" "$final_file"
  completed_names+=("$name")
  digest=$(shasum -a 256 "$final_file" | awk '{print $1}')
  log_line "Completed $name; SHA-256 $digest"
done

(
  cd "$ARCHIVE_DIR" || exit 1
  shasum -a 256 "${completed_names[@]}" > "$CHECKSUM_FILE"
)

CURRENT_NAME=""
CURRENT_PART=""
log_line "All recommended Dudgeon 2013 packages downloaded and verified."
log_line "Archives: $ARCHIVE_DIR"
log_line "Checksums: $CHECKSUM_FILE"
log_line "No archives were extracted automatically."
