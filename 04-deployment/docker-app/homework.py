#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import os
import math
import sys


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


year = int(sys.argv[1])
month = int(sys.argv[2])

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# ## Q1. Notebook
# 
# We'll start with the same notebook we ended up with in homework 1.
# We cleaned it a little bit and kept only the scoring part. You can find the initial notebook [here](homework/starter.ipynb).
# 
# Run this notebook for the March 2023 data.
# 
# What's the standard deviation of the predicted duration for this dataset?
# 
# * 1.24
# * 6.24
# * 12.28
# * 18.28

std_pred = round(np.std(y_pred),2)
print(f"What's the standard deviation of the predicted duration for this dataset?: {std_pred}")


# ## Q2. Preparing the output
# 
# Like in the course videos, we want to prepare the dataframe with the output. 
# 
# First, let's create an artificial `ride_id` column:
# 
# ```python
# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
# ```
# 
# Next, write the ride id and the predictions to a dataframe with results. 
# 
# Save it as parquet:
# 
# ```python
# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )
# ```
# 
# What's the size of the output file?
# 
# * 36M
# * 46M
# * 56M
# * 66M

df_result = pd.DataFrame()
df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df_result['y_predict'] = y_pred


file_path = f"output_file_{year}_{month}.parquet"

df_result.to_parquet(
    file_path,
    engine='pyarrow',
    compression=None,
    index=False
)


# Get the file size in bytes
file_size = os.path.getsize(file_path)

# Convert the size to a more readable format (e.g., KB, MB)
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

# Print the file size
print(f"File size: {convert_size(file_size)}")


# ## Q3. Creating the scoring script
# 
# Now let's turn the notebook into a script. 
# 
# Which command you need to execute for that?

print(f"jupyter nbconvert --to script filename.ipynb")


# ## Q4. Virtual environment
# 
# Now let's put everything into a virtual environment. We'll use pipenv for that.
# 
# Install all the required libraries. Pay attention to the Scikit-Learn version: it should be the same as in the starter
# notebook.
# 
# After installing the libraries, pipenv creates two files: `Pipfile`
# and `Pipfile.lock`. The `Pipfile.lock` file keeps the hashes of the
# dependencies we use for the virtual env.
# 
# What's the first hash for the Scikit-Learn dependency?

print(f"What's the first hash for the Scikit-Learn dependency? ")


# ## Q5. Parametrize the script
# 
# Let's now make the script configurable via CLI. We'll create two 
# parameters: year and month.
# 
# Run the script for April 2023. 
# 
# What's the mean predicted duration? 
# 
# * 7.29
# * 14.29
# * 21.29
# * 28.29
# 
# Hint: just add a print statement to your script.

mean_pred = np.mean(y_pred)
print(f"What's the mean predicted duration?: {mean_pred}")