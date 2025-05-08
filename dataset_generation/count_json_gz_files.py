import json
import gzip
import glob

paths = [
    'database/simlingo_v2_2025_01_10/commentary',
    'database/simlingo_v2_2025_01_10/dreamer',
    'database/simlingo_v2_2025_01_10/drivelm',
]


for path in paths:
    # Get all .json.gz files in all subdirectories
    files = glob.glob(f'{path}/**/*.json.gz', recursive=True)
    # Count the number of files
    print(f'Found {len(files)} files in {path}')
    
