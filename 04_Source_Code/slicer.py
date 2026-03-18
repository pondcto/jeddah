import os
import rasterio
from rasterio.windows import Window
import numpy as np

# ================= CONFIGURATION =================
# Input and Output directories
INPUT_FOLDER  = "Input_Files"
OUTPUT_FOLDER = "Output_Patches"

# Machine Learning Tile Size (256x256)
TILE_SIZE = 256  
# =================================================

# Ensure output directory exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("🚀 STARTING AUTOMATED SLICING PIPELINE...")
print(f"📂 Input Directory:  {INPUT_FOLDER}")
print(f"📂 Output Directory: {OUTPUT_FOLDER}")
print("-" * 60)

# Scan for TIFF files
files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.tif')]

if not files:
    print("❌ ERROR: No .tif files found in Input_Files directory!")
    print("   Please place your GeoTIFFs into the 'Input_Files' folder first.")
else:
    for filename in files:
        src_path = os.path.join(INPUT_FOLDER, filename)
        
        # Create a sub-directory for each file to keep outputs organized
        # Example: Output_Patches/Jeddah_2018_Raw/...
        folder_name = os.path.splitext(filename)[0]
        save_dir = os.path.join(OUTPUT_FOLDER, folder_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print(f"🔨 Processing: {filename}...")
        
        try:
            with rasterio.open(src_path) as src:
                count = 0
                
                # Iterate using a sliding window approach
                # Step size = TILE_SIZE (Non-overlapping tiles)
                for x in range(0, src.width, TILE_SIZE):
                    for y in range(0, src.height, TILE_SIZE):
                        
                        # Calculate window width/height (handling edge cases)
                        width = min(TILE_SIZE, src.width - x)
                        height = min(TILE_SIZE, src.height - y)
                        
                        # FILTER 1: Strict Dimension Check
                        # Ensure the tile is exactly 256x256 (Discard incomplete edge tiles)
                        if width == TILE_SIZE and height == TILE_SIZE:
                            
                            window = Window(x, y, width, height)
                            data = src.read(window=window)
                            
                            # FILTER 2: NoData / Empty Tile Removal
                            # Only save if the tile contains data (max value > 0)
                            if np.nanmax(data) > 0:
                                
                                # Define output filename
                                out_name = f"patch_{x}_{y}.tif"
                                out_path = os.path.join(save_dir, out_name)
                                
                                # Update metadata for the specific patch (preserve georeferencing)
                                profile = src.profile.copy()
                                profile.update({
                                    "driver": "GTiff",
                                    "height": height,
                                    "width": width,
                                    "transform": src.window_transform(window),
                                    "compress": "lzw" # LZW Compression to save disk space
                                })
                                
                                # Write the patch to disk
                                with rasterio.open(out_path, "w", **profile) as dst:
                                    dst.write(data)
                                count += 1
                                
                print(f"   ✅ Success! {count} patches generated in /{folder_name}")

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

print("-" * 60)
print("🎉 BATCH PROCESSING COMPLETE! Please check the output folder.")