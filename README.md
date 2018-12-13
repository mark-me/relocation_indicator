# Graydon Moving Indicator 

Based on historical company data, the goal of the project is to build machine learning models that predicts the probability of a company moving in the future

The structure of the project is the following

### Modules
    - cleaning_merging.py
    - aggregate_transform.py
    - common.py

### (A) Cleaning_Merging
    - cleaning_merging.ipynb
    1) Reading in monthly files delivered by graydon with a custom date range
    2) Making missing values uniform
    3) Renaming of the columns to replace spaces with underscores and lowercase
    4) Combine features with targets
    5) Merging to output yearly files where the key is date_month, id_company, id_branch. 
    6) The yearly files should be named [lower_bound_date]_[upper_bound_date]_clean_merge.csv
    7) Save the files locally and (if possible) to the bucket

### (B) Aggregate_Transform
    - aggregate_transform.ipynb
    1) Reads in output of (A.7) 
    2) Aggregates and transforms data into yearly values where the key is the id_company and id_branch
    3) The yearly aggregated files should be named [lower_bound_date]_[upper_bound_date]_aggregated.csv
    4) Save the files locally and (if possible) to the bucket

### (C) Modelling
    Contains ipython notebooks for modeling the aggregated and merged data. 
    The filenames indicate the type of model.

### (D) Exploration
    Contains ipython notebooks for exploring the data.  
    The baseline probability is computed from compute_baseline_probability_yearly.ipynb.
    
### GC Bucket structure
    - graydon-data
      - 01_input: Will have monthly csv uncleaned files. One per month from 2008 to 2018 (oct)
      - 02_cleaned_merged: Will have the output of A.7
      - 03_aggregated: Will have the output of B.4
