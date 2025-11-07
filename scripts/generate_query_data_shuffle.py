import random
import time
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd
import os, sys
import pyarrow.compute as pc
from typing import List, Optional

def gen_keys_to_get(
    table: pa.Table,
    key_column_name: str,
    split: float = 0.1,
    sampling_rate: float = 0.1,
    fixed_total_sample_size: Optional[int] = None # Nuovo argomento
) -> tuple:
    """
    Generate keys and values from an already-loaded Parquet table.
    The total sample size is now determined by fixed_total_sample_size, 
    if provided, and the specific keys are selected based on the current seed (via random.shuffle).

    Args:
        table: Pre-loaded PyArrow Table containing the data
        key_column_name: Name of the column containing the precomputed keys
        split: Fraction of keys to use for remove and insert operations (0 to 0.5)
        sampling_rate: Fraction of rows to sample (used only if fixed_total_sample_size is None)
        fixed_total_sample_size: The exact number of keys (get+remove+insert) to sample.

    Returns:
        tuple: (keys_to_get, keys_to_remove, keys_to_insert, values_to_insert)
    """
    keys_to_get = []
    keys_to_remove = []
    keys_to_insert = []
    values_to_insert = []

    schema = table.schema
    try:
        key_index = schema.get_field_index(key_column_name)
        content_index = schema.get_field_index("content")
    except Exception as e:
        raise RuntimeError(f"Column not found: {e}")

    if key_index == -1 or content_index == -1:
        raise RuntimeError(f"Required columns not found")

    start_time = time.time()

    key_array = table.column(key_index).to_pylist()
    content_array = table.column(content_index).to_pylist()
    
    # ----------------------------------------------------
    # MODIFICA: Combinazione, filtro dei None, shuffle e taglio deterministico
    # ----------------------------------------------------
    combined = list(zip(key_array, content_array))
    # Rimuove le righe con chiavi nulle
    combined = [(k, v) for k, v in combined if k is not None] 

    # Lo shuffle e controllato da random.seed(current_seed), garantendo set diversi per seed diversi
    random.shuffle(combined)
    
    if fixed_total_sample_size is not None and fixed_total_sample_size > 0:
        # Applica il taglio fisso: numero determinato di chiavi totali
        combined = combined[:fixed_total_sample_size]
        print(f"Sampling fixed size: {len(combined)} keys.")
    elif sampling_rate < 1.0:
        # Campionamento basato su frazione (comportamento precedente se non specificata la dimensione fissa)
        target_size = int(len(combined) * sampling_rate)
        combined = combined[:target_size]
        print(f"Sampling fractional size: {len(combined)} keys (rate={sampling_rate}).")
    else:
        # Usa tutto (se sampling_rate=1.0 e fixed_total_sample_size e None/0)
        print(f"Using all {len(combined)} keys.")
        
    sampled_keys, sampled_values = zip(*combined) if combined else ([], [])
    
    # ----------------------------------------------------
    # Il resto della logica di split per get/remove/insert
    # ----------------------------------------------------
    total_keys = len(sampled_keys)
    split_point1 = int(total_keys * split)
    split_point2 = total_keys - split_point1

    keys_to_remove = list(sampled_keys[:split_point1])
    keys_to_get = list(sampled_keys[split_point1:split_point2])
    keys_to_insert = list(sampled_keys[split_point2:])
    values_to_insert = list(sampled_values[split_point2:])

    print(f"Key extraction time (uniform): {time.time() - start_time:.2f}s")
    print(f"Total sampled keys: {total_keys}. Get: {len(keys_to_get)}, Update: {len(keys_to_remove)}/{len(keys_to_insert)}")

    if len(keys_to_remove) != len(keys_to_insert) or len(keys_to_insert) != len(values_to_insert):
        raise RuntimeError("Mismatch in lengths of keys and values (uniform)")

    return keys_to_get, keys_to_remove, keys_to_insert, values_to_insert


def gen_keys_to_get_zipf(
    table: pa.Table,
    key_column_name: str,
    alpha: float = 1.0,
    fixed_sample_size: Optional[int] = None # Nuovo argomento
) -> List:
    """
    Generates a list of keys from a PyArrow Table based on Zipfian distributed indices,
    with a deterministic size if fixed_sample_size is provided.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    try:
        key_column = table.column(key_column_name)
    except pa.lib.ArrowKeyError:
        raise RuntimeError(f"Column '{key_column_name}' not found")

    key_array = key_column.to_numpy()
    valid_mask = ~pa.compute.is_null(key_column).to_numpy()
    filtered_keys = key_array[valid_mask].tolist()
    num_unique_keys = len(filtered_keys)

    if num_unique_keys == 0:
        return []

    if alpha <= 1:
        print(f"WARNING: alpha ({alpha}) is <= 1. numpy.random.zipf requires alpha > 1 for a true Zipfian distribution. Results might not be as expected.")

    # ----------------------------------------------------
    # MODIFICA: Determina la dimensione dell'array Zipfian
    # ----------------------------------------------------
    if fixed_sample_size is not None and fixed_sample_size > 0:
        target_size = fixed_sample_size
        print(f"Generating Zipfian distribution of fixed size: {target_size}.")
    else:
        target_size = num_unique_keys # Comportamento di default: genera un indice per ogni chiave univoca
        print(f"Generating Zipfian distribution of size: {target_size} (equal to unique keys).")
        
    # Generate Zipfian distributed indices (controllato da np.random.seed)
    zipf_indices = np.random.zipf(alpha, size=target_size) - 1

    # Use modulo to ensure indices are within the bounds of the keys
    bounded_indices = zipf_indices % num_unique_keys
    
    # Select keys based on the generated indices
    sampled_keys = [filtered_keys[i] for i in bounded_indices]
    
    # Stampa la dimensione finale
    print(f"Zipfian sampled keys length: {len(sampled_keys)}")

    return sampled_keys


def create_parquet_without_keys(
    input_table: pa.Table,
    keys_to_exclude: List[str],
    key_column_name: str,
    output_parquet_file: Optional[str] = None
) -> pa.Table:
    """
    Creates a new Parquet table (and optionally file) without rows where keys are in keys_to_exclude.
    Optimized for large datasets using vectorized operations.

    Args:
        input_table: Pre-loaded input PyArrow Table
        keys_to_exclude: List of full keys to exclude (including line numbers)
        key_column_name: Name of the column containing the keys
        output_parquet_file: Optional path to write output (if None, returns table only)

    Returns:
        Filtered PyArrow Table

    Note:
        Assumes all keys are distinct and key_column_name exists
    """
    start_time = time.time()
    # Convert exclusion list to a set for O(1) lookups
    exclude_set = set(keys_to_exclude)

    # Vectorized operation to create mask
    key_array = input_table.column(key_column_name)
    mask = pc.is_in(key_array, value_set=pa.array(exclude_set, type=key_array.type))
    mask = pc.invert(mask)

    # Apply filter
    filtered_table = input_table.filter(mask)
    
    print(f"Filtering time: {time.time() - start_time:.2f}s")

    # Optional write to file
    if output_parquet_file:
        try:
            pq.write_table(filtered_table, output_parquet_file, compression=None)
            print(f"Saved filtered Parquet to: {output_parquet_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to write output Parquet file: {e}")

    return filtered_table


if __name__ == '__main__':
    if len(sys.argv) != 1 + 4:
        print("Usage: python generate_query_data.py <parquet_file> <filepath_column_name> <probability> <seed>")
        print("Example: python generate_query_data.py /data/file.parquet max_stars_repo_path 0.1 1234")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    key_column_name = sys.argv[2]
    probability = float(sys.argv[3])
    
    try:
        current_seed = int(sys.argv[4])
    except ValueError:
        raise ValueError("Seed must be an integer")
        
    if probability < 0 or probability > 1:
        raise ValueError("Probability must be between 0 and 1")

    # Set random seeds for reproducibility, using the provided seed
    print(f"Using seed: {current_seed}")
    random.seed(current_seed)
    np.random.seed(current_seed)
    # ----------------------------------------------------
    
    # Load the original Parquet file
    try:
        table = pq.read_table(parquet_file)
    except Exception as e:
        raise RuntimeError(f"Failed to read Parquet file: {e}")

    split = probability 
    sampling_rate = 1.0 
    alpha = 1.5
    
    fixed_total_sample_size = 100 

    schema = table.schema
    if key_column_name not in schema.names:
        raise RuntimeError(f"Column '{key_column_name}' not found in the Parquet file")
    if "content" not in schema.names:
        raise RuntimeError("Column 'content' not found in the Parquet file")
    if table.num_rows == 0:
        raise RuntimeError("The Parquet file is empty")

    print(f"Parquet file '{parquet_file}' loaded successfully.")
    print(f"Number of rows in the Parquet file: {table.num_rows}.")

    # Uniform sampling (ora con dimensione fissa e seed-dipendente)
    ktgs, ktrs, ktis, vtis = gen_keys_to_get(
        table,
        key_column_name,
        split=split,
        sampling_rate=sampling_rate, # Usato solo se fixed_total_sample_size e None
        fixed_total_sample_size=fixed_total_sample_size
    )

    # Zipf sampling (ora con dimensione fissa e seed-dipendente)
    ktgs_zipf = gen_keys_to_get_zipf(
        table,
        key_column_name,
        alpha=alpha,
        fixed_sample_size=fixed_total_sample_size
    )
        
    # Get base path without extension
    base_path = os.path.splitext(parquet_file)[0]

    # Salvataggio dei risultati
    
    # Save keys_to_get (uniform sampling)
    df_ktgs = pd.DataFrame({'keys_to_get': ktgs})
    outfilepath = f"{base_path}-s{current_seed}.get-{probability}-{sampling_rate}.parquet"
    df_ktgs.to_parquet(outfilepath, compression=None)
    print(f"Saved keys_to_get (uniform) to {outfilepath} with {len(df_ktgs)} lines")

    # Save keys_to_get (zipf sampling)
    df_ktgs_zipf = pd.DataFrame({'keys_to_get': ktgs_zipf})
    outfilepath = f"{base_path}-s{current_seed}.getzipf-{probability}-{sampling_rate}-{alpha}.parquet"
    df_ktgs_zipf.to_parquet(outfilepath, compression=None)
    print(f"Saved keys_to_get_zipf to {outfilepath} with {len(df_ktgs_zipf)} lines")

    # Save update data (uniform sampling)
    df_update = pd.DataFrame({
        'keys_to_remove': ktrs,
        'keys_to_insert': ktis,
        'values_to_insert': vtis
    })
    outfilepath = f"{base_path}-s{current_seed}.update-{probability}-{sampling_rate}.parquet"
    df_update.to_parquet(outfilepath, compression=None)
    print(f"Saved update data to {outfilepath} with {len(df_update)} lines")
    
    # Nota: Ho aggiornato i nomi dei file di output per includere il seed e la dimensione per chiarezza.

    print(f"Done with base path: {base_path}")
    