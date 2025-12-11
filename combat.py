import os
import numpy as np
import pandas as pd
import zarr
import shutil
from tqdm import tqdm
from neuroCombat import neuroCombat
from canvas.model.data.imc_dataset import CANVASDatasetWithLocation
import argparse

def main():
    # Initialize paths
    # source_data_path = '/gpfs/data/proteomics/home/bm3772/canvas_examples/optimal_channels/data/processed_data/data'
    # output_data_path = '/gpfs/data/proteomics/home/bm3772/canvas_examples/optimal_channels_combat/data/processed_data/data'
    # source_data_path = '/gpfs/data/proteomics/data/Cervical_mIF/output/data'
    # output_data_path = '/gpfs/data/proteomics/data/Cervical_mIF/output/combat'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    
    source_data_path = args.data_path
    output_data_path = args.output_path
    
    # Create output directory
    os.makedirs(output_data_path, exist_ok=True)
    
    # Find all .er directories
    er_directories = []
    for root, dirs, files in os.walk(source_data_path):
        for dir_name in dirs:
            if dir_name.endswith('.er'):
                er_directories.append(os.path.join(root, dir_name))
    
    print(f"Found {len(er_directories)} .er directories")
    
    # Step 1: Collect all tile data for ComBat
    print("Step 1: Collecting tile data from all slides...")
    all_slide_data = collect_all_tile_data(er_directories)
    
    if not all_slide_data:
        print("No valid data found!")
        return
    
    # Step 2: Run ComBat normalization
    print("Step 2: Running ComBat normalization...")
    normalized_data = run_combat_normalization(all_slide_data)
    
    if normalized_data is None:
        print("ComBat normalization failed!")
        return
    
    # Step 3: Save normalized data back to CANVAS format
    print("Step 3: Saving normalized data in CANVAS format...")
    save_normalized_data(normalized_data, all_slide_data, output_data_path)
    
    print("ComBat normalization pipeline completed successfully!")

def collect_all_tile_data(er_directories):
    """
    Collect tile data from all slides for ComBat normalization
    """
    all_slide_data = []
    
    for er_dir in tqdm(er_directories, desc="Collecting data"):
        slide_id = os.path.basename(er_dir).replace('.er', '')
        
        # Check required files
        zarr_file = os.path.join(er_dir, "data.zarr")
        positions_file = os.path.join(er_dir, "tiles", "positions_128.csv")
        channels_file = os.path.join(er_dir, "channels.csv")
        
        if not all(os.path.exists(f) for f in [zarr_file, positions_file, channels_file]):
            print(f"Missing required files for {slide_id}, skipping...")
            continue
        
        try:
            # Read channel information
            channels_df = pd.read_csv(channels_file)
            available_channels = channels_df['marker'].tolist()
            
            # Use first 3 channels or all available
            # common_channel_names = available_channels[:3] if len(available_channels) >= 3 else available_channels
            
            # Use ALL available channels:
            common_channel_names = available_channels

            # Create CANVAS dataset
            dataset = CANVASDatasetWithLocation(
                root_path=er_dir,
                tile_size=128,
                tiles_dir="tiles",
                common_channel_names=common_channel_names,
                transform=None,
                lazy=True
            )
            
            # Collect tile data
            slide_tiles = []
            tile_positions = []
            
            for i in range(len(dataset)):
                try:
                    image, (sample_label, location) = dataset[i]
                    
                    # Convert to numpy if needed
                    if hasattr(image, 'numpy'):
                        image = image.numpy()
                    
                    if len(image.shape) == 3:
                        slide_tiles.append(image)
                        tile_positions.append(location)
                    
                except Exception as e:
                    print(f"Error processing tile {i} in {slide_id}: {e}")
                    continue
            
            if slide_tiles:
                slide_data = {
                    'slide_id': slide_id,
                    'er_dir': er_dir,
                    'tiles': slide_tiles,
                    'positions': tile_positions,
                    'channels': common_channel_names,
                    'channels_df': channels_df
                }
                all_slide_data.append(slide_data)
                print(f"Collected {len(slide_tiles)} tiles from {slide_id}")
            
        except Exception as e:
            print(f"Error processing {slide_id}: {e}")
            continue
    
    return all_slide_data

def run_combat_normalization(all_slide_data):
    """
    Run ComBat normalization on collected tile data
    """
    print("Preparing data for ComBat...")
    
    # Combine all tiles and create batch labels
    all_tiles = []
    batch_labels = []
    slide_indices = []
    
    for slide_idx, slide_data in enumerate(all_slide_data):
        for tile in slide_data['tiles']:
            all_tiles.append(tile)
            batch_labels.append(slide_idx)  # Use slide index as batch
            slide_indices.append(slide_idx)
    
    print(f"Total tiles: {len(all_tiles)}")
    print(f"Number of slides (batches): {len(all_slide_data)}")
    
    # Convert to numpy array
    all_tiles_array = np.stack(all_tiles)  # Shape: (num_tiles, channels, height, width)
    num_tiles, num_channels, height, width = all_tiles_array.shape
    
    print(f"Data shape: {all_tiles_array.shape}")
    
    # Reshape for ComBat: (features, samples)
    # Each pixel in each channel is a feature, each tile is a sample
    features_per_tile = num_channels * height * width
    combat_data = all_tiles_array.reshape(num_tiles, features_per_tile).T
    
    print(f"ComBat input shape: {combat_data.shape}")
    
    # Create batch dataframe
    batch_df = pd.DataFrame({'batch': batch_labels})
    
    try:
        # Run ComBat normalization
        print("Running ComBat normalization...")
        combat_result = neuroCombat(
            dat=combat_data,
            covars=batch_df,
            batch_col='batch'
        )
        
        # Get normalized data
        normalized_data = combat_result['data']
        
        # Reshape back to original format
        normalized_data = normalized_data.T.reshape(num_tiles, num_channels, height, width)
        
        print(f"ComBat normalization completed!")
        print(f"Normalized data shape: {normalized_data.shape}")
        
        return {
            'normalized_tiles': normalized_data,
            'slide_indices': slide_indices,
            'original_data': all_slide_data
        }
        
    except Exception as e:
        print(f"ComBat normalization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_normalized_data(normalized_data, all_slide_data, output_data_path):
    """
    Save normalized data back to CANVAS-compatible format
    """
    normalized_tiles = normalized_data['normalized_tiles']
    slide_indices = normalized_data['slide_indices']
    original_data = normalized_data['original_data']
    
    # Group normalized tiles back by slide
    tile_idx = 0
    for slide_idx, slide_data in enumerate(original_data):
        slide_id = slide_data['slide_id']
        num_tiles = len(slide_data['tiles'])
        
        print(f"Processing slide {slide_id} ({num_tiles} tiles)")
        
        # Get normalized tiles for this slide
        slide_normalized_tiles = normalized_tiles[tile_idx:tile_idx + num_tiles]
        tile_idx += num_tiles
        
        # Create output directory for this slide
        output_slide_dir = os.path.join(output_data_path, f"{slide_id}.er")
        os.makedirs(output_slide_dir, exist_ok=True)
        
        # Reconstruct the full zarr from normalized tiles
        reconstructed_zarr = reconstruct_zarr_from_tiles(
            slide_normalized_tiles, 
            slide_data['positions'], 
            slide_data['er_dir']
        )
        
        if reconstructed_zarr is not None:
            # Save normalized zarr
            output_zarr_path = os.path.join(output_slide_dir, "data.zarr")
            zarr.save(output_zarr_path, reconstructed_zarr)
            print(f"Saved normalized zarr: {output_zarr_path}")
            
            # Copy other required files
            copy_supporting_files(slide_data['er_dir'], output_slide_dir)
        else:
            print(f"Failed to reconstruct zarr for {slide_id}")

def reconstruct_zarr_from_tiles(normalized_tiles, positions, source_er_dir):
    """
    Reconstruct full zarr array from normalized tiles
    """
    try:
        # Load original zarr to get dimensions
        original_zarr = zarr.open(os.path.join(source_er_dir, "data.zarr"), mode='r')
        channels, height, width = original_zarr.shape
        
        # Create new zarr array with same dimensions
        reconstructed = np.copy(original_zarr[:])
        
        # Place normalized tiles back into the zarr
        tile_size = 128
        for i, (tile_data, position) in enumerate(zip(normalized_tiles, positions)):
            h_coord, w_coord = position
            
            # Check bounds
            if (h_coord + tile_size <= height and w_coord + tile_size <= width and
                h_coord >= 0 and w_coord >= 0):
                
                # Apply transpose to fix diagonal reflection for zarr storage
                transformed_tile = np.copy(tile_data)
                for c in range(transformed_tile.shape[0]):
                    # Transpose each channel to fix diagonal reflection
                    transformed_tile[c] = np.transpose(transformed_tile[c], (1, 0))
                
                # Place transformed tile back into zarr
                reconstructed[:, h_coord:h_coord+tile_size, w_coord:w_coord+tile_size] = transformed_tile
                
        return reconstructed
        
    except Exception as e:
        print(f"Error reconstructing zarr: {e}")
        return None

def copy_supporting_files(source_dir, output_dir):
    """
    Copy channels.csv and tiles directory to output
    """
    try:
        # Copy channels.csv
        source_channels = os.path.join(source_dir, "channels.csv")
        output_channels = os.path.join(output_dir, "channels.csv")
        if os.path.exists(source_channels):
            shutil.copy2(source_channels, output_channels)
            print(f"Copied channels.csv")
        
        # Copy tiles directory
        source_tiles = os.path.join(source_dir, "tiles")
        output_tiles = os.path.join(output_dir, "tiles")
        if os.path.exists(source_tiles):
            shutil.copytree(source_tiles, output_tiles, dirs_exist_ok=True)
            print(f"Copied tiles directory")
        
    except Exception as e:
        print(f"Error copying supporting files: {e}")

if __name__ == '__main__':
    main()