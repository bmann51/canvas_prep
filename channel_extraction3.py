#!/usr/bin/env python3
"""
Combined Channel Extraction + ROI Cropping + Multi-Region Splitting
Extract specified channels and crop to ROI regions, splitting disconnected regions
FIXED VERSION - Preserves all 9 channels for spatial visualization compatibility
"""

import zarr
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import json
import argparse
import sys
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import DBSCAN

class ChannelROIExtractor:
    def __init__(self, 
                 source_base: str,
                 target_base: str,
                 target_channels: List[str],
                 images: List[str],
                 dataset_name: Optional[str] = None,
                 max_gap: int = 256,
                 padding: int = 0,
                 preserve_all_channels: bool = False):  # Changed default to False
        """
        Initialize the combined channel and ROI extractor
        
        Args:
            source_base: Path to source data directory
            target_base: Path to target output directory  
            target_channels: List of channel names to extract (for training/analysis)
            images: List of image names to process
            dataset_name: Optional name for the dataset
            max_gap: Maximum gap between tiles to consider them connected (pixels)
            padding: Padding around each ROI region (pixels)
            preserve_all_channels: If True, preserve all 9 channels in output for visualization
        """
        
        self.source_base = source_base
        self.target_base = target_base
        self.target_channels = target_channels
        self.images = images
        self.dataset_name = dataset_name or f"{len(target_channels)}_channel_roi_dataset"
        self.max_gap = max_gap
        self.padding = padding
        self.preserve_all_channels = preserve_all_channels
        
        # Create full target path with dataset name
        self.full_target_base = str(Path(target_base) / self.dataset_name)
        
        print(f"Initialized ChannelROIExtractor")
        print(f"Source: {self.source_base}")
        print(f"Target: {self.full_target_base}")
        print(f"Target channels for analysis: {self.target_channels}")
        print(f"Preserve all channels for visualization: {self.preserve_all_channels}")
        if not self.preserve_all_channels:
            print(f"Will extract ONLY the {len(self.target_channels)} specified channels")
        print(f"Images to process: {len(self.images)}")
        print(f"Max gap for connectivity: {self.max_gap}px")
        print(f"Padding: {self.padding}px")
    
    @classmethod
    def from_config_file(cls, config_path: str):
        """Create extractor from JSON config file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(
            source_base=config['source_base'],
            target_base=config['target_base'],
            target_channels=config['target_channels'],
            images=config['images'],
            dataset_name=config.get('dataset_name'),
            max_gap=config.get('max_gap', 256),
            padding=config.get('padding', 0),
            preserve_all_channels=config.get('preserve_all_channels', True)
        )
    
    def find_channel_indices(self, channels_csv_path: str) -> Tuple[Dict, Dict, List]:
        """Find the indices of target channels and get all available channels"""
        
        channels_df = pd.read_csv(channels_csv_path)
        
        channel_indices = {}
        channel_names = {}
        
        print(f"  Channel file columns: {channels_df.columns.tolist()}")
        print(f"  Channel file shape: {channels_df.shape}")
        
        # Determine which column contains the marker names
        marker_column = None
        for col in ['marker', 'channel_name', 'name', 'protein']:
            if col in channels_df.columns:
                marker_column = col
                break
        
        if marker_column is None:
            print(f"  ✗ Could not find marker column in channels file")
            return {}, {}, []
        
        print(f"  Using '{marker_column}' column for marker names")
        available_markers = channels_df[marker_column].tolist()
        print(f"  Available markers ({len(available_markers)}): {available_markers}")
        
        # Build mapping for target channels
        for target in self.target_channels:
            found = False
            print(f"  Looking for: '{target}'")
            
            for idx, row in channels_df.iterrows():
                marker_name = str(row[marker_column]).strip()
                channel_idx = row.get('channel', idx)
                
                print(f"    Checking: '{marker_name}' (channel {channel_idx})")
                
                # Flexible matching: exact, contains, or partial match
                if (target.lower() == marker_name.lower() or 
                    target.lower() in marker_name.lower() or
                    marker_name.lower() in target.lower()):
                    
                    channel_indices[target] = channel_idx
                    channel_names[target] = marker_name
                    found = True
                    print(f"    ✓ MATCH: {target} -> {marker_name} (zarr channel {channel_idx})")
                    break
            
            if not found:
                print(f"    ✗ No match found for '{target}'")
                print(f"        Available options: {available_markers}")
        
        print(f"  TARGET CHANNEL MAPPING: {channel_indices}")
        
        # Get all available channels in order
        all_channels_ordered = []
        for idx, row in channels_df.iterrows():
            channel_idx = row.get('channel', idx)
            marker_name = str(row[marker_column]).strip()
            all_channels_ordered.append((channel_idx, marker_name))
        
        # Sort by channel index to maintain order
        all_channels_ordered.sort(key=lambda x: x[0])
        all_channel_names = [name for _, name in all_channels_ordered]
        
        print(f"  ALL CHANNELS IN ORDER: {all_channel_names}")
        
        return channel_indices, channel_names, all_channel_names
    
    def find_roi_clusters(self, positions: List[Tuple[int, int]], tile_size: int = 128) -> List[List[Tuple[int, int]]]:
        """
        Find separate ROI clusters using DBSCAN clustering
        """
        if not positions:
            return []
        
        positions_array = np.array(positions)
        
        # Use DBSCAN to find clusters
        # eps is the maximum distance between tiles to be in same cluster
        eps = self.max_gap + tile_size
        clustering = DBSCAN(eps=eps, min_samples=1).fit(positions_array)
        
        # Group positions by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(positions[i])
        
        return list(clusters.values())
    
    def extract_roi_region(self, source_zarr: zarr.Array, region_positions: List[Tuple[int, int]], 
                          channel_indices: Dict[str, int], all_channel_names: List[str], 
                          tile_size: int = 128) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Extract a single ROI region - REVERTED to working version, only fix coordinates
        """
        # Calculate bounding box for this region (ORIGINAL LOGIC THAT WORKED)
        h_coords = [pos[0] for pos in region_positions]
        w_coords = [pos[1] for pos in region_positions]
        
        orig_height, orig_width = source_zarr.shape[1], source_zarr.shape[2]
        
        min_h = max(0, min(h_coords) - self.padding)
        max_h = min(orig_height, max(h_coords) + tile_size + self.padding)
        min_w = max(0, min(w_coords) - self.padding)
        max_w = min(orig_width, max(w_coords) + tile_size + self.padding)
        
        print(f"  Extracting ROI region: h={min_h}:{max_h}, w={min_w}:{max_w}")
        print(f"  Source zarr shape: {source_zarr.shape}")

        if self.preserve_all_channels:
            # PRESERVE ALL CHANNELS - extract the entire zarr for spatial visualization compatibility
            print(f"  Preserving all {source_zarr.shape[0]} channels for visualization compatibility")
            roi_zarr = source_zarr[:, min_h:max_h, min_w:max_w]
            print(f"  Extracted all channels, shape: {roi_zarr.shape}")
        else:
            # EXTRACT ONLY TARGET CHANNELS - for training/analysis workflows
            print(f"  Extracting only target channels: {self.target_channels}")
            print(f"  Channel indices: {channel_indices}")
            
            extracted_channels = []
            
            for target_name in self.target_channels:
                if target_name in channel_indices:
                    source_idx = channel_indices[target_name]
                    print(f"  Extracting {target_name} from zarr channel {source_idx}")
                    
                    try:
                        channel_data = source_zarr[source_idx, min_h:max_h, min_w:max_w]
                        extracted_channels.append(channel_data)
                        print(f"  ✓ Successfully extracted {target_name}, shape: {channel_data.shape}")
                    except Exception as e:
                        print(f"  ✗ Error extracting {target_name} from index {source_idx}: {e}")
                        raise
                else:
                    print(f"  ✗ Channel {target_name} not found in channel_indices!")
                    raise ValueError(f"Channel {target_name} not found")
            
            if len(extracted_channels) != len(self.target_channels):
                raise ValueError(f"Expected {len(self.target_channels)} channels, got {len(extracted_channels)}")
            
            # Stack channels in correct order
            roi_zarr = np.stack(extracted_channels, axis=0)
            print(f"  Final ROI zarr shape: {roi_zarr.shape}")
        
        # Update positions to be actual pixel coordinates within the cropped zarr
        roi_positions = []
        for h, w in region_positions:
            # Convert to relative tile indices first
            rel_h = h - min_h  
            rel_w = w - min_w
            # Then convert to actual pixel coordinates within the ROI
            pixel_h = rel_h * tile_size  # Convert tile index to pixel coordinate
            pixel_w = rel_w * tile_size  # Convert tile index to pixel coordinate
            roi_positions.append((pixel_h, pixel_w))
        
        print(f"  Updated {len(roi_positions)} positions to pixel coordinates")
        if roi_positions:
            print(f"  Example positions: {roi_positions[:3]}...")  # Show first 3 for debugging
        
        return roi_zarr, roi_positions
    
    def create_output_structure(self):
        """Create the output directory structure"""
        
        target_path = Path(self.full_target_base)
        target_path.mkdir(parents=True, exist_ok=True)
        
        data_path = target_path / "data" / "processed_data" / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        
        analysis_path = target_path / "analysis"
        analysis_path.mkdir(exist_ok=True)
        
        print(f"✓ Created output structure at: {target_path}")
        return data_path, analysis_path
    
    def process_image(self, image_name: str, data_path: Path) -> List[str]:
        """Process a single image, extracting channels and splitting ROI regions"""
        
        print(f"\n{'='*50}")
        print(f"Processing: {image_name}")
        print(f"{'='*50}")
        
        # Source paths
        source_dir = Path(self.source_base) / image_name
        source_zarr_path = source_dir / "data.zarr"
        source_channels = source_dir / "channels.csv"
        positions_file = source_dir / "tiles" / "positions_128.csv"
        
        # Validation
        required_files = [source_zarr_path, source_channels, positions_file]
        for file_path in required_files:
            if not file_path.exists():
                print(f"✗ Required file not found: {file_path}")
                return []
        
        # Find channel indices and get all available channels
        channel_indices, channel_names, all_channel_names = self.find_channel_indices(source_channels)
        
        # Validation for target channels (only if not preserving all channels)
        if not self.preserve_all_channels and len(channel_indices) != len(self.target_channels):
            print(f"✗ Could not find all target channels.")
            print(f"  Requested: {self.target_channels}")
            print(f"  Found: {list(channel_indices.keys())}")
            print(f"  Missing: {[ch for ch in self.target_channels if ch not in channel_indices]}")
            return []
        
        if self.preserve_all_channels:
            print(f"Preserving all {len(all_channel_names)} channels: {all_channel_names}")
        else:
            print(f"Target channel mapping:")
            for target, actual in channel_names.items():
                print(f"  {target} -> {actual} (index {channel_indices[target]})")
        
        # Load positions
        positions_df = pd.read_csv(positions_file)
        if 'h_coord' in positions_df.columns and 'w_coord' in positions_df.columns:
            positions = list(zip(positions_df['h_coord'], positions_df['w_coord']))
        else:
            positions = list(zip(positions_df.iloc[:, 0], positions_df.iloc[:, 1]))
        
        if not positions:
            print(f"✗ No ROI positions found")
            return []
        
        # Find ROI clusters
        roi_clusters = self.find_roi_clusters(positions)
        print(f"Found {len(roi_clusters)} separate ROI regions")
        
        # Load source zarr
        source_zarr = zarr.open(str(source_zarr_path), mode='r')
        print(f"Source zarr shape: {source_zarr.shape}")
        
        successful_outputs = []
        
        # Process each ROI region
        for region_idx, region_positions in enumerate(roi_clusters):
            print(f"\nProcessing region {region_idx + 1}/{len(roi_clusters)} ({len(region_positions)} tiles)")
            
            try:
                # Extract ROI region
                roi_zarr, roi_positions = self.extract_roi_region(
                    source_zarr, region_positions, channel_indices, all_channel_names
                )
                
                # Determine output name
                if len(roi_clusters) > 1:
                    base_name = image_name.replace('.er', '')
                    output_name = f"{base_name}_part{region_idx + 1}.er"
                else:
                    output_name = image_name
                
                # Create output directory
                output_dir = data_path / output_name
                output_dir.mkdir(exist_ok=True)
                
                # Save zarr
                output_zarr_path = output_dir / "data.zarr"
                zarr.save(str(output_zarr_path), roi_zarr)
                
                print(f"  ✓ Saved zarr: {roi_zarr.shape} -> {output_zarr_path}")
                
                # Create channels.csv in correct format matching working data
                if self.preserve_all_channels:
                    # Copy the original channels.csv to maintain all channels
                    shutil.copy2(source_channels, output_dir / "channels.csv")
                    print(f"  ✓ Preserved original channels.csv with all channels")
                else:
                    # Create new channels.csv with target channels in correct format
                    new_channels_data = []
                    for i, target_name in enumerate(self.target_channels):
                        new_channels_data.append({
                            'channel': i,
                            'marker': channel_names[target_name],
                            'channel_name': target_name  # Include channel_name column to match working format
                        })
                    
                    new_channels_df = pd.DataFrame(new_channels_data)
                    channels_csv_path = output_dir / "channels.csv"
                    new_channels_df.to_csv(channels_csv_path, index=False)
                    print(f"  ✓ Created new channels.csv with {len(self.target_channels)} target channels")
                
                # Save updated positions in correct format
                tiles_dir = output_dir / "tiles"
                tiles_dir.mkdir(exist_ok=True)
                
                # Create positions DataFrame with correct column names and format
                positions_data = []
                for i, (h, w) in enumerate(roi_positions):
                    positions_data.append({' ': i, 'h': h, 'w': w})
                
                positions_df_new = pd.DataFrame(positions_data)
                positions_csv_path = tiles_dir / "positions_128.csv"
                positions_df_new.to_csv(positions_csv_path, index=False)
                
                # Copy other files from tiles directory (except positions_128.csv)
                source_tiles_dir = source_dir / "tiles"
                if source_tiles_dir.exists():
                    for item in source_tiles_dir.iterdir():
                        if item.name != "positions_128.csv":
                            target_item = tiles_dir / item.name
                            if item.is_file():
                                shutil.copy2(item, target_item)
                            elif item.is_dir():
                                shutil.copytree(item, target_item, dirs_exist_ok=True)
                
                # Copy other metadata files
                for file_name in ['thumbnails', 'metadata.json']:
                    source_file = source_dir / file_name
                    if source_file.exists():
                        target_file = output_dir / file_name
                        if source_file.is_dir():
                            shutil.copytree(source_file, target_file, dirs_exist_ok=True)
                        else:
                            shutil.copy2(source_file, target_file)
                
                # Save extraction metadata
                extraction_info = {
                    'original_image': image_name,
                    'region_index': region_idx,
                    'total_regions': len(roi_clusters),
                    'original_shape': list(source_zarr.shape),
                    'roi_shape': list(roi_zarr.shape),
                    'num_tiles': len(roi_positions),
                    'preserve_all_channels': self.preserve_all_channels,
                    'target_channels': self.target_channels,
                    'all_channels': all_channel_names,
                    'channel_mapping': {target: channel_names.get(target, target) for target in self.target_channels} if not self.preserve_all_channels else 'all_preserved',
                    'extraction_timestamp': str(pd.Timestamp.now())
                }
                
                metadata_file = output_dir / "extraction_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(extraction_info, f, indent=2)
                
                successful_outputs.append(output_name)
                print(f"  ✓ Successfully created: {output_name}")
                
            except Exception as e:
                print(f"  ✗ Error processing region {region_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return successful_outputs
    
    def create_common_channels_file(self, data_path: Path, all_channel_names: List[str]):
        """Create common_channels.txt file"""
        
        common_channels_file = data_path / "common_channels.txt"
        
        if self.preserve_all_channels:
            # Write all channels to common_channels.txt for visualization compatibility
            with open(common_channels_file, 'w') as f:
                for channel in all_channel_names:
                    f.write(f"{channel}\n")
            print(f"✓ Created common_channels.txt with all {len(all_channel_names)} channels: {all_channel_names}")
        else:
            # Write only target channels
            with open(common_channels_file, 'w') as f:
                for channel in self.target_channels:
                    f.write(f"{channel}\n")
            print(f"✓ Created common_channels.txt with target channels: {self.target_channels}")
        
        return common_channels_file
    
    def run_extraction(self) -> int:
        """Main extraction workflow"""
        
        print(f"\n{'='*60}")
        print(f"CHANNEL + ROI EXTRACTION STARTED: {self.dataset_name}")
        print(f"{'='*60}")
        
        # Create output structure
        data_path, analysis_path = self.create_output_structure()
        
        # Save configuration
        config = {
            'source_base': self.source_base,
            'target_base': self.target_base,
            'target_channels': self.target_channels,
            'images': self.images,
            'dataset_name': self.dataset_name,
            'max_gap': self.max_gap,
            'padding': self.padding,
            'preserve_all_channels': self.preserve_all_channels
        }
        
        config_file = analysis_path / "extraction_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Process images
        all_successful_outputs = []
        processing_summary = {}
        all_channel_names = None
        
        for image in tqdm(self.images, desc="Processing images"):
            try:
                successful_outputs = self.process_image(image, data_path)
                all_successful_outputs.extend(successful_outputs)
                processing_summary[image] = {
                    'success': len(successful_outputs) > 0,
                    'output_count': len(successful_outputs),
                    'output_names': successful_outputs
                }
                
                # Get channel names from first successful image
                if all_channel_names is None and successful_outputs:
                    source_dir = Path(self.source_base) / image
                    source_channels = source_dir / "channels.csv"
                    if source_channels.exists():
                        _, _, all_channel_names = self.find_channel_indices(source_channels)
                
            except Exception as e:
                print(f"✗ Error processing {image}: {e}")
                processing_summary[image] = {
                    'success': False,
                    'error': str(e)
                }
                continue
        
        # Create outputs
        if all_successful_outputs and all_channel_names:
            self.create_common_channels_file(data_path, all_channel_names)
        
        # Create summary report
        summary_report = {
            'extraction_summary': {
                'dataset_name': self.dataset_name,
                'target_channels': self.target_channels,
                'preserve_all_channels': self.preserve_all_channels,
                'input_images': len(self.images),
                'output_images': len(all_successful_outputs),
                'processing_details': processing_summary
            },
            'settings': {
                'max_gap': self.max_gap,
                'padding': self.padding,
                'extraction_timestamp': str(pd.Timestamp.now())
            }
        }
        
        report_file = analysis_path / "extraction_report.json"
        with open(report_file, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        # Create readable summary
        summary_file = analysis_path / "extraction_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Channel + ROI Extraction Summary: {self.dataset_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Target channels for analysis ({len(self.target_channels)}): {', '.join(self.target_channels)}\n")
            f.write(f"Preserve all channels for visualization: {self.preserve_all_channels}\n")
            f.write(f"Input images: {len(self.images)}\n")
            f.write(f"Output images: {len(all_successful_outputs)}\n")
            f.write(f"Max gap for connectivity: {self.max_gap}px\n")
            f.write(f"Padding: {self.padding}px\n\n")
            
            f.write("Processing results:\n")
            for input_img, details in processing_summary.items():
                if details['success']:
                    f.write(f"  ✓ {input_img} -> {details['output_count']} region(s)\n")
                    for output_name in details['output_names']:
                        f.write(f"    - {output_name}\n")
                else:
                    f.write(f"  ✗ {input_img} - Failed\n")
            
            f.write(f"\nOutput location: {self.full_target_base}/data/processed_data/data/\n")
            f.write(f"Next step: sbatch canvas_training_parameterized.sh {self.dataset_name}\n")
        
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Input images: {len(self.images)}")
        print(f"Output images: {len(all_successful_outputs)}")
        print(f"Target channels: {', '.join(self.target_channels)}")
        print(f"Preserve all channels: {self.preserve_all_channels}")
        print(f"Output location: {self.full_target_base}")
        
        return len(all_successful_outputs)


def create_example_config():
    """Create an example configuration file"""
    
    example_config = {
        "dataset_name": "9channel_roi_dataset",
        "source_base": "/gpfs/data/proteomics/home/bm3772/canvas_examples/all_images_9channels/data/processed_data/data",
        "target_base": "/gpfs/data/proteomics/home/bm3772/canvas_examples",
        "target_channels": [
            "E-cadherin", "CD31", "CD3e"
        ],
        "images": [
            "20250305-Jharna-34933-A1_Scan1.er",
            "20250225-Jharna-02433-A1_Scan1.er", 
            "20250305-Jharna-09002-A1_Scan1.er",
            "20250318-Jharna-28873-A1_Scan1.er"
        ],
        "max_gap": 256,
        "padding": 0,
        "preserve_all_channels": true
    }
    
    with open("example_roi_config.json", 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print("✓ Created example_roi_config.json")


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description="Extract channels and crop to ROI regions")
    
    # Configuration options
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--create-example", action="store_true", help="Create example config file")
    
    # Direct configuration options
    parser.add_argument("--source", help="Source data directory")
    parser.add_argument("--target", help="Target output directory")
    parser.add_argument("--channels", nargs="+", help="Channel names to extract")
    parser.add_argument("--images", nargs="+", help="Image names to process")
    parser.add_argument("--all-er", action="store_true", help="Process all .er directories in source")
    parser.add_argument("--dataset-name", help="Dataset name")
    parser.add_argument("--max-gap", type=int, default=256, help="Max gap between connected tiles (default: 256)")
    parser.add_argument("--padding", type=int, default=0, help="Padding around ROI regions (default: 0)")
    parser.add_argument("--preserve-all-channels", action="store_true", default=True, help="Preserve all 9 channels for visualization (default: True)")
    parser.add_argument("--extract-only-target", action="store_true", help="Extract only target channels (disables preserve-all-channels)")
    
    args = parser.parse_args()
    
    # Create example config
    if args.create_example:
        create_example_config()
        return
    
    # Handle preserve_all_channels logic
    preserve_all = args.preserve_all_channels and not args.extract_only_target  # Default to False unless explicitly set
    
    # Load from config file
    if args.config:
        extractor = ChannelROIExtractor.from_config_file(args.config)
    
    # Load from command line arguments
    elif all([args.source, args.target, args.channels]) and (args.images or args.all_er):
        
        # Handle --all-er flag
        if args.all_er:
            print(f"Scanning for all .er directories in: {args.source}")
            source_path = Path(args.source)
            all_er_dirs = []
            
            for item in source_path.iterdir():
                if item.is_dir() and item.name.endswith('.er'):
                    all_er_dirs.append(item.name)
            
            if not all_er_dirs:
                print(f"✗ No .er directories found in {args.source}")
                return
            
            print(f"Found {len(all_er_dirs)} .er directories")
            images_to_process = all_er_dirs
        else:
            images_to_process = args.images
        
        extractor = ChannelROIExtractor(
            source_base=args.source,
            target_base=args.target,
            target_channels=args.channels,
            images=images_to_process,
            dataset_name=args.dataset_name,
            max_gap=args.max_gap,
            padding=args.padding,
            preserve_all_channels=preserve_all
        )
    
    else:
        print("✗ Either provide --config file or required arguments:")
        print("  --source, --target, --channels, and either --images or --all-er")
        print("Run with --create-example to create a template config file")
        return
    
    # Run extraction
    success_count = extractor.run_extraction()
    
    if success_count > 0:
        print(f"\n🎯 Successfully created dataset with {success_count} images!")
        print(f"💡 All 9 channels preserved for spatial visualization compatibility")
    else:
        print(f"\n❌ Extraction failed - check error messages above")


if __name__ == "__main__":
    main()