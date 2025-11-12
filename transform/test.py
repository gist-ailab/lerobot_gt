import pyarrow.parquet as pq
import numpy as np

if __name__ == "__main__":


 # Read the Parquet file
   # table = pq.read_table('/HDD/gr00t-custom-fine-tune/transform/CTB_lerobot_dataset/data/chunk-000/episode_000000.parquet')
   #  print(table)
   #  print(action)
   # df = table.to_pandas()
   # print(df)

   table = pq.read_table('./so100/data/chunk-000/file-000.parquet')
   df = table.to_pandas()
   # print(df)
   print(df.columns)
   print(df[452:480])
  #  df.to_csv("so100_output.csv", index=False)


   table = pq.read_table('./CTB_lerobot_dataset_3.0/data/chunk-000/file-000.parquet')
   df = table.to_pandas()
   # print(df)
  #  print(df.columns)
  #  print(df[250:280])
  #  df.to_csv("ours_output.csv", index=False)



   table = pq.read_table('./so100/meta/episodes/chunk-000/file-000.parquet')
   df = table.to_pandas()
  #  print(df['dataset_to_index'])
  #  print(df['dataset_from_index'])
   print(df.columns)
   print(df[1:2])
   for k, v in df[1:2].items():
       print(k, v)
  #  df.to_csv("output.csv", index=False)

   table = pq.read_table('./CTB_lerobot_dataset_3.0/meta/episodes/chunk-000/file-000.parquet')
   df = table.to_pandas()
  #  print(df['dataset_to_index'])
  #  print(df['dataset_from_index'])
   print(df.columns)
   print(df[1:2])
  #  df.to_csv("output_.csv", index=False)

  #  table = pq.read_table('./so100/meta/tasks.parquet')
  #  df = table.to_pandas()
  # #  print(df['dataset_to_index'])
  # #  print(df['dataset_from_index'])
  #  print(df.columns)
  #  print(df)

  #  table = pq.read_table('./CTB_lerobot_dataset_3.0/meta/tasks.parquet')
  #  df = table.to_pandas()
  # #  print(df['dataset_to_index'])
  # #  print(df['dataset_from_index'])
  #  print(df.columns)
  #  print(df)


  #  table = pq.read_table('./CTB_lerobot_dataset_3/meta/episodes/chunk-000/file-000.parquet')
  #  df = table.to_pandas()
  #  print(df['dataset_to_index'])
  #  print(df['dataset_from_index'])
  #  print(df.columns)