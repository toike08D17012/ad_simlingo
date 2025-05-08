import glob
import gzip
import ujson
import os
import tqdm
import multiprocessing
from functools import partial


def check_json_file(json_file):
    try:
        with gzip.open(json_file, 'rt') as fin:
            ujson.load(fin)
        return None  # File is valid
    except Exception as e:
        print(f'Error loading {json_file}: {e}')
        return json_file  # Return file path if it should be deleted


def main():
    path = '/mnt/lustre/work/geiger/krenz73/coding/07_simlingo/simlingo_cleanup/database/simlingo_v2_2025_01_10/commentary'
    path = '/mnt/lustre/work/geiger/krenz73/coding/07_simlingo/simlingo_cleanup/database/simlingo_v2_2025_01_10/dreamer'
    all_jsons = glob.glob(f'{path}/**/*.json.gz', recursive=True)

    print(f'Found {len(all_jsons)} json files')
    
    # Use multiprocessing pool to process files in parallel
    num_cores = multiprocessing.cpu_count()
    print(f'Using {num_cores} cores for processing')
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm.tqdm(
            pool.imap(check_json_file, all_jsons),
            total=len(all_jsons)
        ))
    
    # Filter out None results
    to_delete = [file for file in results if file is not None]

    print(f'Deleting {len(to_delete)} json files')

    breakpoint()
    # Uncomment to actually delete files
    for json_file in to_delete:
        os.remove(json_file)
        print(f'Deleted {json_file}')


if __name__ == "__main__":
    main()