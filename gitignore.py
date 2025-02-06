
import pyarrow as pa
import pyarrow.dataset as ds
import json
import os
from pathlib import Path

def verify_arrow_file(file_path):
    """Verify if file is valid Arrow format"""
    try:
        with open(file_path, 'rb') as f:
            # Check Arrow magic number
            magic = f.read(6)
            return magic == b'ARROW1'
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False

def read_dataset(base_dir):
    """Read dataset with validation"""
    base_path = Path(base_dir)
    
    # Find files
    arrow_file = base_path / 'data-00000-of-00001.arrow'
    dataset_info = base_path / 'dataset_info.json'
    state_json = base_path / 'state.json'
    
    # Verify files exist
    if not all(f.exists() for f in [arrow_file, dataset_info, state_json]):
        raise FileNotFoundError(f"Missing required files in {base_dir}")
    
    # Verify Arrow format
    if not verify_arrow_file(arrow_file):
        raise ValueError(f"Invalid Arrow file format: {arrow_file}")
        
    # Read files
    reader = pa.ipc.RecordBatchFileReader(str(arrow_file))
    data = reader.read_all()
    
    with open(dataset_info) as f:
        info = json.load(f)
        
    with open(state_json) as f:
        state = json.load(f)
        
    return {
        'data': data,
        'info': info,
        'state': state
    }

if __name__ == "__main__":
    try:
        dataset_path = "experiment/ppo-cot-rho1b-gsm/episodes/episodes_0000.json"
        print(f"Reading from: {dataset_path}")
        dataset = read_dataset(dataset_path)
        print(f"Successfully loaded dataset")
    except Exception as e:
        print(f"Error: {e}")