"""
data_loading.py

Purpose: Download play-by-play data from nflfastR for seasons 2021-2024
Author: Anokh Palakurthi
Date: 2025-11-04

This script:
1. Imports the nfl_data_py library (wrapper for nflfastR)
2. Downloads play-by-play data for 2021-2024 seasons
3. Saves it locally as a CSV file for future use
"""

# =============================================================================
# IMPORTS
# =============================================================================

import nfl_data_py as nfl  # For downloading NFL data
import pandas as pd         # For data manipulation
import os                   # For file path handling

print("Starting data loading process...")
print("=" * 60)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define which seasons to download
SEASONS = [2021, 2022, 2023, 2024]

# Define where to save the data
DATA_DIR = "../data/raw" 
OUTPUT_FILE = os.path.join(DATA_DIR, "pbp_2021_2024.csv")

print(f"Downloading seasons: {SEASONS}")
print(f"Will save to: {OUTPUT_FILE}")
print("=" * 60)

# =============================================================================
# DOWNLOAD DATA
# =============================================================================

print("\nDownloading play-by-play data from nflfastR...")

try:
    # Download play-by-play data for specified seasons
    pbp = nfl.import_pbp_data(years=SEASONS)
    
    print(f"Successfully downloaded {len(pbp):,} plays")
    print(f"   Seasons: {pbp['season'].unique()}")
    print(f"   Columns: {len(pbp.columns)} variables")
    
except Exception as e:
    print(f"Error downloading data: {e}")
    raise

# =============================================================================
# SAVE DATA LOCALLY
# =============================================================================

print("\nSaving data to local file...")

try:
    # Create directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save as CSV
    pbp.to_csv(OUTPUT_FILE, index=False)
    
    # Get file size for confirmation
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    
    print(f"   Data saved successfully!")
    print(f"   Location: {OUTPUT_FILE}")
    print(f"   File size: {file_size_mb:.1f} MB")
    
except Exception as e:
    print(f" Error saving data: {e}")
    raise

# =============================================================================
# PREVIEW DATA
# =============================================================================

print("\n" + "=" * 60)
print("DATA PREVIEW")
print("=" * 60)

# Show basic info
print(f"\nDataset shape: {pbp.shape[0]:,} rows × {pbp.shape[1]} columns")
print(f"\nFirst few rows:")
print(pbp.head())

print(f"\nColumn names (first 20):")
print(pbp.columns.tolist()[:20])

print("\n" + "=" * 60)
print("  DATA LOADING COMPLETE!")
print("=" * 60)