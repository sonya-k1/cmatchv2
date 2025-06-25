import json
import pandas as pd
import os

def extract_data(match_result):
    """
    Extracts overall score, part scores, and positions from the match result,
    preserving repeated part information.

    Returns tuple of:
    - overall_score
    - parts_data (list of dictionaries: [{'name': '...', 'score': ..., 'start': ..., 'end': ...}])
    - errors (if any)
    """
    result = json.loads(match_result)

    if not result or not isinstance(result, dict):
        return 0, [], None, None
    overall_score = result.get("score", 0)
    errors = result.get("errors")
    parts_data = []

    if "path" in result and result["path"]:
        for part in result["path"]:
            part_info = {
                "name": part.get("name", ""),
                "score": part.get("score", 0),
                "start": part.get("start", 0),
                "end": part.get("end", 0)
            }
            parts_data.append(part_info)
    overlap_bases = result.get("overlap")

    return overall_score, parts_data, errors, overlap_bases

def store_cmatch_results(results_json, parquet_file, similarity_threshold, overlap_tolerance):
    """
    Stores match function output in a Parquet file with additional parameters,
    handling repeated parts by storing them in lists.
    If the file exists, appends the new data to it.

    :param results_json: JSON string from match function
    :param parquet_file: Path to save the output Parquet file
    :param similarity_threshold: Similarity threshold value
    :param overlap_tolerance: Overlap tolerance value
    :return: Number of records processed
    """
    try:
        
        data = json.loads(results_json)[0]

        if not data:
            print("Warning: Empty results data")
            return 0

        records = []
        for item in data:
            target = item.get("target", "UNKNOWN")  # Sequence ID
            reconstruct = item.get("reconstruct", "")  # Reconstruction pattern
        
            # Extract scores, positions and errors
            overall_score, parts_data, errors, overlap_bases = extract_data(json.dumps(item))
            

            record = {
                "ID": target,
                "Reconstruction": reconstruct,
                "Overall_Score": overall_score,
                "Parts": parts_data,  # Store the list of part dictionaries
                "Errors": str(errors) if errors is not None else None,
                "Breaking_Overlap": overlap_bases,
                "Similarity_Threshold": similarity_threshold,
                "Overlap_Tolerance": overlap_tolerance
            }
            records.append(record)

        
        new_df = pd.DataFrame(records)


        if os.path.exists(parquet_file):
            try:
               
                existing_df = pd.read_parquet(parquet_file)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)

                # Remove duplicates based on ID and parameters
                combined_df = combined_df.drop_duplicates(
                    subset=["ID", "Similarity_Threshold", "Overlap_Tolerance"],
                    keep="last"
                )
                combined_df.to_parquet(parquet_file, index=False)
                print(f"Added {len(new_df)} records to existing file {parquet_file}. Total records: {len(combined_df)}")
            except Exception as e:
                print(f"Error appending to existing file: {e}")
                print(f"Creating new file with current data only.")
                new_df.to_parquet(parquet_file, index=False)
        else:
            # File doesn't exist, create new
            new_df.to_parquet(parquet_file, index=False)
            print(f"Created new file {parquet_file} with {len(new_df)} records.")

        return len(new_df)

    except Exception as e:
        print(f"Error processing match results: {e}")
        return 0


if __name__ == "__main__":
    # Sample JSON 
    example_json = '''[
  {
    "target":"pb_gfp_99_sub_only_S_1_0",
    "reconstruct":"J23101_B0030_combined-GFP-B0015",
    "score":0.989034203871302,
    "path":[
      {
        "name":"J23101_B0030_combined",
        "score":1.0,
        "start":0,
        "length":50,
        "end":50
      },
      {
        "name":"GFP",
        "score":0.9674620390455532,
        "start":50,
        "length":922,
        "end":972
      },
      {
        "name":"B0015",
        "score":1.0,
        "start":972,
        "length":129,
        "end":1101
      }
    ],
    "errors":null
  }
]'''
    
    store_cmatch_results(example_json, "test_results.parquet", 0.8, 0.2)