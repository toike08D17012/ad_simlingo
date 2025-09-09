import subprocess
import tarfile
from pathlib import Path

from tqdm import tqdm

qa_list = [
    'commentary_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_001.tar.gz',
    'commentary_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_002.tar.gz',
    'commentary_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_003.tar.gz',
]

commentry_list = [
    'drivelm_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_001.tar.gz',
    'drivelm_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_002.tar.gz',
    'drivelm_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_003.tar.gz',
]

data_list = [
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_001.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_002.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_003.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_004.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_005.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_006.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_007_part1.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_007_part2.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_008.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_009.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_010.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_011.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_012.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_013.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_014.tar.gz',
    'data_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_015.tar.gz',
]

dreamer_list = [
    'dreamer_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_001.tar.gz',
    'dreamer_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_002.tar.gz',
    'dreamer_simlingo_validation_3_scenarios_routes_validation_random_weather_seed_4_balanced_100_chunk_003.tar.gz',
]

download_file_list = qa_list + commentry_list + dreamer_list + data_list

with open('downloaded_list.txt') as f:
    downloaded_files = f.read().split('\n')

for file_path in tqdm(download_file_list):
    if file_path in downloaded_files:
        print(f'{file_path} is already installed, therefore skipping download.')
        continue
    subprocess.run(f'wget https://huggingface.co/datasets/RenzKa/simlingo/resolve/main/{file_path}', shell=True)
    with tarfile.open(file_path) as f:
        f.extractall()
    subprocess.run(f'rm {file_path}', shell=True)
    with open('downloaded_list.txt', 'a') as f:
        f.write(file_path + '\n')
    print(f'Download and extraction is just complete: {file_path}')
