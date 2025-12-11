#!/usr/bin/env python3
"""
Configurable multichannel image channel extractor
Extract any specified channels from raw multichannel images
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
from typing import List, Dict, Optional

class ConfigurableChannelExtractor:
    def __init__(self, 
                 source_base: str,
                 target_base: str,
                 target_channels: List[str],
                 images: List[str],
                 dataset_name: Optional[str] = None):
        """
        Initialize the configurable channel extractor
        
        Args:
            source_base: Path to source data directory
            target_base: Path to target output directory  
            target_channels: List of channel names to extract
            images: List of image names to process
            dataset_name: Optional name for the dataset (used in output paths)
        """
        
        self.source_base = source_base
        self.target_base = target_base
        self.target_channels = target_channels
        self.images = images
        self.dataset_name = dataset_name or f"{len(target_channels)}_channel_dataset"
        
        # Create full target path with dataset name
        self.full_target_base = str(Path(target_base) / self.dataset_name)
        
        print(f"Initialized ConfigurableChannelExtractor")
        print(f"Source: {self.source_base}")
        print(f"Target: {self.full_target_base}")
        print(f"Target channels: {self.target_channels}")
        print(f"Images to process: {len(self.images)}")
        print(f"Dataset name: {self.dataset_name}")
    
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
            dataset_name=config.get('dataset_name')
        )
    
    def save_config(self, output_path: str):
        """Save current configuration to JSON file"""
        config = {
            'source_base': self.source_base,
            'target_base': self.target_base,
            'target_channels': self.target_channels,
            'images': self.images,
            'dataset_name': self.dataset_name
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration saved to: {output_path}")
    
    def list_available_channels(self, sample_image: Optional[str] = None):
        """
        List all available channels from a sample image
        
        Args:
            sample_image: Image name to check, uses first image if None
        """
        if sample_image is None:
            sample_image = self.images[0]
        
        channels_csv = Path(self.source_base) / sample_image / "channels.csv"
        
        if not channels_csv.exists():
            print(f"✗ Channels file not found: {channels_csv}")
            return []
        
        channels_df = pd.read_csv(channels_csv)
        
        print(f"\nAvailable channels in {sample_image}:")
        print(f"{'Index':<6} {'Channel':<10} {'Marker':<20}")
        print("-" * 50)
        
        available_channels = []
        for idx, row in channels_df.iterrows():
            channel_idx = row.get('channel', idx)
            marker = row.get('marker', 'Unknown')
            channel_name = row.get('channel_name', marker)
            
            print(f"{channel_idx:<6} {channel_name:<10} {marker:<20}")
            available_channels.append({
                'index': channel_idx,
                'name': channel_name,
                'marker': marker
            })
        
        return available_channels
    
    def find_channel_indices(self, channels_csv_path: str) -> tuple:
        """Find the indices of target channels with flexible matching"""
        
        channels_df = pd.read_csv(channels_csv_path)
        
        channel_indices = {}
        channel_names = {}
        
        print(f"  Channel file columns: {channels_df.columns.tolist()}")
        
        # Determine which column contains the marker names
        marker_column = None
        for col in ['marker', 'channel_name', 'name', 'protein']:
            if col in channels_df.columns:
                marker_column = col
                break
        
        if marker_column is None:
            print(f"  ✗ Could not find marker column in channels file")
            return {}, {}
        
        print(f"  Using '{marker_column}' column for marker names")
        print(f"  Available markers: {channels_df[marker_column].tolist()}")
        
        for target in self.target_channels:
            found = False
            
            for idx, row in channels_df.iterrows():
                marker_name = str(row[marker_column]).strip()
                channel_idx = row.get('channel', idx)  # Use 'channel' column or row index
                
                # Flexible matching: exact, contains, or partial match
                if (target.lower() == marker_name.lower() or 
                    target.lower() in marker_name.lower() or
                    marker_name.lower() in target.lower()):
                    
                    channel_indices[target] = channel_idx
                    channel_names[target] = marker_name
                    found = True
                    print(f"  ✓ Found {target} -> {marker_name} (zarr channel {channel_idx})")
                    break
            
            if not found:
                print(f"  ✗ Warning: Could not find channel matching '{target}'")
        
        return channel_indices, channel_names
    
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
    
    def extract_channels_from_image(self, image_name: str, data_path: Path) -> bool:
        """Extract the specified channels from a single image"""
        
        print(f"\n{'='*50}")
        print(f"Processing: {image_name}")
        print(f"{'='*50}")
        
        # Source paths
        source_dir = Path(self.source_base) / image_name
        source_zarr = source_dir / "data.zarr"
        source_channels = source_dir / "channels.csv"
        
        # Validation
        if not source_zarr.exists():
            print(f"✗ Source zarr not found: {source_zarr}")
            return False
        
        if not source_channels.exists():
            print(f"✗ Source channels.csv not found: {source_channels}")
            return False
        
        # Find channel indices
        channel_indices, channel_names = self.find_channel_indices(source_channels)
        
        if len(channel_indices) != len(self.target_channels):
            print(f"✗ Could not find all target channels.")
            print(f"  Requested: {self.target_channels}")
            print(f"  Found: {list(channel_indices.keys())}")
            return False
        
        print(f"Channel mapping:")
        for target, actual in channel_names.items():
            print(f"  {target} -> {actual} (index {channel_indices[target]})")
        
        # Create target directory
        target_dir = data_path / image_name
        target_dir.mkdir(exist_ok=True)
        
        # Load source zarr
        source_store = zarr.open(str(source_zarr), mode='r')
        n_channels, height, width = source_store.shape
        
        print(f"Source: {source_store.shape} {source_store.dtype}")
        
        # Create target zarr
        target_zarr = target_dir / "data.zarr"
        target_store = zarr.open(
            str(target_zarr), 
            mode='w',
            shape=(len(self.target_channels), height, width),
            chunks=(1, source_store.chunks[1], source_store.chunks[2]),
            dtype=source_store.dtype,
            compressor=source_store.compressor
        )
        
        print(f"Target: {target_store.shape} {target_store.dtype}")
        
        # Copy channels in the order specified
        channel_mapping = []
        for new_idx, target_name in enumerate(self.target_channels):
            source_idx = channel_indices[target_name]
            actual_name = channel_names[target_name]
            
            print(f"Copying {actual_name} (source {source_idx} -> target {new_idx})")
            target_store[new_idx, :, :] = source_store[source_idx, :, :]
            
            channel_mapping.append({
                'new_index': new_idx,
                'old_index': source_idx,
                'target_name': target_name,
                'actual_name': actual_name
            })
        
        # Create new channels.csv with consistent format
        new_channels_data = []
        for i, target_name in enumerate(self.target_channels):
            new_channels_data.append({
                'channel': i,
                'marker': channel_names[target_name],
                'channel_name': target_name
            })
        
        new_channels_df = pd.DataFrame(new_channels_data)
        target_channels_csv = target_dir / "channels.csv"
        new_channels_df.to_csv(target_channels_csv, index=False)
        
        # Copy other files
        other_files = ['tiles', 'thumbnails', 'metadata.json']
        for file_name in other_files:
            source_file = source_dir / file_name
            target_file = target_dir / file_name
            
            if source_file.exists():
                if source_file.is_dir():
                    if target_file.exists():
                        shutil.rmtree(target_file)
                    shutil.copytree(source_file, target_file)
                    print(f"✓ Copied directory: {file_name}")
                else:
                    shutil.copy2(source_file, target_file)
                    print(f"✓ Copied file: {file_name}")
        
        # Save extraction log
        extraction_log = {
            'image': image_name,
            'source_shape': list(source_store.shape),
            'target_shape': list(target_store.shape),
            'target_channels': self.target_channels,
            'channel_mapping': channel_mapping,
            'extraction_timestamp': str(pd.Timestamp.now())
        }
        
        log_file = target_dir / "extraction_log.json"
        with open(log_file, 'w') as f:
            json.dump(extraction_log, f, indent=2)
        
        print(f"✓ Successfully extracted {len(self.target_channels)} channels")
        return True
    
    def create_common_channels_file(self, data_path: Path):
        """Create common_channels.txt file"""
        
        common_channels_file = data_path / "common_channels.txt"
        with open(common_channels_file, 'w') as f:
            for channel in self.target_channels:
                f.write(f"{channel}\n")
        
        print(f"✓ Created common_channels.txt with: {self.target_channels}")
        return common_channels_file
    
    def create_summary_report(self, analysis_path: Path, successful_images: List[str]):
        """Create extraction summary report"""
        
        report = {
            'extraction_summary': {
                'dataset_name': self.dataset_name,
                'target_channels': self.target_channels,
                'total_images_requested': len(self.images),
                'successful_extractions': len(successful_images),
                'failed_extractions': len(self.images) - len(successful_images),
                'successful_images': successful_images,
                'failed_images': [img for img in self.images if img not in successful_images]
            },
            'dataset_info': {
                'output_location': self.full_target_base,
                'channels_per_image': len(self.target_channels),
                'ready_for_canvas': True,
                'recommended_command': f"sbatch canvas_training_parameterized.sh {self.dataset_name}"
            },
            'configuration': {
                'source_base': self.source_base,
                'target_base': self.target_base,
                'extraction_timestamp': str(pd.Timestamp.now())
            }
        }
        
        # Save JSON report
        report_file = analysis_path / "extraction_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save readable summary
        summary_file = analysis_path / "extraction_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Channel Extraction Summary: {self.dataset_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Target channels ({len(self.target_channels)}): {', '.join(self.target_channels)}\n")
            f.write(f"Images processed: {len(successful_images)}/{len(self.images)}\n")
            f.write(f"Output location: {self.full_target_base}\n\n")
            
            f.write("Successful extractions:\n")
            for img in successful_images:
                f.write(f"  ✓ {img}\n")
            
            if len(successful_images) < len(self.images):
                f.write("\nFailed extractions:\n")
                for img in self.images:
                    if img not in successful_images:
                        f.write(f"  ✗ {img}\n")
            
            f.write(f"\nNext steps:\n")
            f.write(f"1. Verify data at: {self.full_target_base}/data/processed_data/data/\n")
            f.write(f"2. Run CANVAS training: sbatch canvas_training_parameterized.sh {self.dataset_name}\n")
        
        print(f"✓ Summary saved to: {summary_file}")
        return report_file
    
    def run_extraction(self) -> int:
        """Main extraction workflow"""
        
        print(f"\n{'='*60}")
        print(f"CHANNEL EXTRACTION STARTED: {self.dataset_name}")
        print(f"{'='*60}")
        
        # Create output structure
        data_path, analysis_path = self.create_output_structure()
        
        # Save configuration
        config_file = analysis_path / "extraction_config.json"
        self.save_config(str(config_file))
        
        # Process images
        successful_images = []
        
        for image in tqdm(self.images, desc="Processing images"):
            try:
                success = self.extract_channels_from_image(image, data_path)
                if success:
                    successful_images.append(image)
            except Exception as e:
                print(f"✗ Error processing {image}: {e}")
                continue
        
        # Create outputs
        if successful_images:
            self.create_common_channels_file(data_path)
        
        self.create_summary_report(analysis_path, successful_images)
        
        # Final summary
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Successfully processed: {len(successful_images)}/{len(self.images)} images")
        print(f"Channels extracted: {', '.join(self.target_channels)}")
        print(f"Output location: {self.full_target_base}")
        
        if successful_images:
            print(f"\n✓ Dataset ready for CANVAS!")
            print(f"Run: sbatch canvas_training_parameterized.sh {self.dataset_name}")
        else:
            print(f"\n✗ No images successfully processed")
        
        return len(successful_images)


def create_example_config():
    """Create an example configuration file"""
    
    example_config = {
        "dataset_name": "custom_3channel",
        "source_base": "/gpfs/data/proteomics/data/Cervical_mIF/output/data",
        "target_base": "/gpfs/data/proteomics/home/bm3772/canvas_examples",
        "target_channels": ["DAPI", "E-cadherin", "CD163"],
        "images": [
            "20250305-Jharna-34933-A1_Scan1.er",
            "20250225-Jharna-02433-A1_Scan1.er", 
            "20250305-Jharna-09002-A1_Scan1.er",
            "20250318-Jharna-28873-A1_Scan1.er"
        ]
    }
    
    with open("example_config.json", 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print("✓ Created example_config.json")
    print("Edit this file and run: python script.py --config example_config.json")


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description="Extract specific channels from multichannel images")
    
    # Configuration options
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--create-example", action="store_true", help="Create example config file")
    
    # Direct configuration options
    parser.add_argument("--source", help="Source data directory")
    parser.add_argument("--target", help="Target output directory")
    parser.add_argument("--channels", nargs="+", help="Channel names to extract")
    parser.add_argument("--images", nargs="+", help="Image names to process")
    parser.add_argument("--dataset-name", help="Dataset name")
    
    # Utility options
    parser.add_argument("--list-channels", help="List available channels from specified image")
    
    args = parser.parse_args()
    
    # Create example config
    if args.create_example:
        create_example_config()
        return
    
    # List channels utility
    if args.list_channels:
        if not args.source:
            print("✗ --source required when using --list-channels")
            return
        
        extractor = ConfigurableChannelExtractor(
            source_base=args.source,
            target_base="temp",
            target_channels=["temp"],
            images=[args.list_channels]
        )
        extractor.list_available_channels(args.list_channels)
        return
    
    # Load from config file
    if args.config:
        extractor = ConfigurableChannelExtractor.from_config_file(args.config)
    
    # Load from command line arguments
    elif all([args.source, args.target, args.channels, args.images]):
        extractor = ConfigurableChannelExtractor(
            source_base=args.source,
            target_base=args.target,
            target_channels=args.channels,
            images=args.images,
            dataset_name=args.dataset_name
        )
    
    else:
        print("✗ Either provide --config file or all required arguments")
        print("Run with --create-example to create a template config file")
        print("Run with --help to see all options")
        return
    
    # Run extraction
    success_count = extractor.run_extraction()
    
    if success_count > 0:
        print(f"\n🎯 Successfully created dataset with {success_count} images!")
    else:
        print(f"\n❌ Extraction failed - check error messages above")


if __name__ == "__main__":
    main()
