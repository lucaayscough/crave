#!/usr/bin/env python3

import os
import sys
import subprocess
import tempfile
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Convert RAVE TorchScript model to packed CRAVE format by running both export and pack scripts.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Path to input TorchScript file')
    parser.add_argument('-o', '--output', required=True,
                       help='Path to output packed .bin file')
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    export_script_path = script_dir / 'export_rave_weights.py'
    pack_script_path = script_dir / 'pack_rave_weights.py'
    
    # Check if both scripts exist
    if not export_script_path.exists():
        print(f"Error: Export script not found at {export_script_path}", file=sys.stderr)
        return 1
    
    if not pack_script_path.exists():
        print(f"Error: Pack script not found at {pack_script_path}", file=sys.stderr)
        return 1
    
    # Use specified temp directory
    temp_dir = Path(".")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_dir_path = str(temp_dir)
    
    try:
        print(f"Using temporary directory: {temp_dir_path}")
        
        # Step 1: Run the export script
        print("\nStep 1: Exporting RAVE model to individual .bin files...")
        export_cmd = [
            sys.executable, str(export_script_path),
            '-i', args.input,
            '-o', temp_dir_path
        ]
        
        print(f"Running: {' '.join(export_cmd)}")
        result = subprocess.run(export_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running export script:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return 1
        
        print(result.stdout)
        
        # Step 2: Run the pack script
        print("\nStep 2: Packing .bin files into single output file...")
        pack_cmd = [
            sys.executable, str(pack_script_path),
            '-i', temp_dir_path,
            '-o', args.output,
            '--config', str(2),
            '--block_size', str(2048),
            '--num_latents', str(32),
            '--sample_rate', str(44100)
        ]
        
        print(f"Running: {' '.join(pack_cmd)}")
        result = subprocess.run(pack_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running pack script:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return 1
        
        print(result.stdout)
        
        print(f"\nConversion completed successfully!")
        print(f"Output file: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        return 1
        
    finally:
        # Clean up temporary .bin files
        print(f"\nCleaning up temporary .bin files in: {temp_dir_path}")
        for bin_file in Path(temp_dir_path).glob('*.bin'):
            try:
                bin_file.unlink()
            except OSError:
                pass  # Ignore errors during cleanup

if __name__ == "__main__":
    sys.exit(main())
