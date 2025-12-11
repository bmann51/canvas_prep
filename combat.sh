#!/bin/bash
#SBATCH --job-name=combat
#SBATCH --output=logs/%j_combat.out
#SBATCH --error=logs/%j_combat.err
#SBATCH --partition=gpu8_short      # 18GB per CPU
#SBATCH --cpus-per-task=20          # 20 × 18GB = 360GB available
#SBATCH --mem=100G                  # Request 320GB 
#SBATCH --time=8:00:00
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:0


# Default paths (can be overridden by command line arguments)
DATA_PATH="${1:-canvas_examples/all_images_9channels/data/processed_data/data}"
OUTPUT_PATH="${2:-canvas_examples/all_images_9channels_combat/data/processed_data/data}"

# DATA_PATH="${1:-/gpfs/data/proteomics/data/Cervical_mIF/output/data}"
# OUTPUT_PATH="${2:-/gpfs/data/proteomics/data/Cervical_mIF/output/combat}"

echo "Using data path: $DATA_PATH"
echo "Using output path: $OUTPUT_PATH"

# Run ComBat harmonization
python combat.py --data_path "$DATA_PATH" --output_path "$OUTPUT_PATH"