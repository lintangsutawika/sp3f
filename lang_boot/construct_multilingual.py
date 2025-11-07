import os
import argparse
import numpy as np
import pandas as pd


data_path = "/mnt/labshare/lsutawik/lbr/data"
lang_dir = [
    "deepscaler-gpt5nano-ar-q2.5-7b",
    "deepscaler-gpt5nano-bn-q2.5-7b",
    "deepscaler-gpt5nano-de-q2.5-7b",
    "deepscaler-gpt5nano-es-q2.5-7b",
    "deepscaler-gpt5nano-fr-q2.5-7b",
    "deepscaler-gpt5nano-hi-q2.5-7b",
    "deepscaler-gpt5nano-id-q2.5-7b",
    "deepscaler-gpt5nano-it-q2.5-7b",
    "deepscaler-gpt5nano-ja-q2.5-7b",
    "deepscaler-gpt5nano-ko-q2.5-7b",
    "deepscaler-gpt5nano-my-q2.5-7b",
    "deepscaler-gpt5nano-pt-q2.5-7b",
    "deepscaler-gpt5nano-ru-q2.5-7b",
    "deepscaler-gpt5nano-sw-q2.5-7b",
    "deepscaler-gpt5nano-te-q2.5-7b",
    "deepscaler-gpt5nano-th-q2.5-7b",
    "deepscaler-gpt5nano-yo-q2.5-7b",
    "deepscaler-gpt5nano-zh-q2.5-7b",
    "deepscaler-gpt5nano-id-q2.5-7b",
    "deepscaler-gpt5nano-it-q2.5-7b",
    "deepscaler-gpt5nano-ja-q2.5-7b",
    "deepscaler-gpt5nano-ko-q2.5-7b",
    "deepscaler-gpt5nano-my-q2.5-7b",
    "deepscaler-gpt5nano-pt-q2.5-7b",
    "deepscaler-gpt5nano-ru-q2.5-7b",
    "deepscaler-gpt5nano-sw-q2.5-7b",
    "deepscaler-gpt5nano-te-q2.5-7b",
    "deepscaler-gpt5nano-th-q2.5-7b",
    "deepscaler-gpt5nano-yo-q2.5-7b",
    "deepscaler-gpt5nano-zh-q2.5-7b",
]

# Concat all train.parquet files
def concat_all_parquet_files(data_path, lang_dir, file_name="train.parquet"):
    all_dfs = []
    for lang in lang_dir:
        lang_path = os.path.join(data_path, lang, file_name)
        if os.path.exists(lang_path):
            df = pd.read_parquet(lang_path)
            all_dfs.append(df)
        else:
            print(f"File not found: {lang_path}")
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        print("No dataframes to concatenate.")
        return pd.DataFrame()

if __name__ == "__main__":
    output_path = "deepscaler-gpt5nano-all-q2.5-7b"
    for file_name in ["train.parquet", "test.parquet"]:
        combined_df = concat_all_parquet_files(data_path, lang_dir, file_name)
        if not combined_df.empty:
            combined_df.to_parquet(os.path.join(data_path, output_path, file_name), index=False)
        else:
            print(f"No data to save for {file_name}.")