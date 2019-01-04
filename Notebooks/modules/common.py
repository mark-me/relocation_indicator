import sys, os
import gcsfs
from google.cloud import storage
import pandas as pd

class GC_Data_Processing(object):

    def __init__(self, name_project, name_bucket):
        """Returns a new Google Cloud Bucket object."""
        cs = storage.Client()
        self.name_bucket = name_bucket
        self.gc_fs = gcsfs.GCSFileSystem(project=name_project)
        self.gc_bucket = cs.get_bucket(name_bucket)

    def get_gc_bucket(self):
        return self.gc_bucket

    def get_gc_fs(self):
        return self.gc_fs

    def local_file_to_bucket(self, file_source, dir_bucket):
        """Uploads a file to the Google Cloud Bucket."""
        file_destination = dir_bucket + "/" + os.path.basename(file_source)
        blob = self.gc_bucket.blob(file_destination)
        blob.upload_from_filename(file_source)
        print('File {} uploaded to {}.'.format(
            file_source,
            file_destination))

    def get_df_from_bucket(self, file_path, sep = ",", 
                           index_col = None, dtype = None, na_values = None,
                           parse_dates = False):
        with self.gc_fs.open(self.name_bucket + "/" + file_path) as f:
            df = pd.read_csv(f)
        return(df)

    def save_df_locally(self, df, dir_output, file_name, as_json = False):
        """ Saves df as json or csv locally on server """
        if not os.path.exists(dir_output):
            os.mkdir(dir_output)
        file_path = dir_output + '/' + file_name
        if as_json:        
            df.to_json(file_path)
        else:
            df.to_csv(file_path, float_format='%.6f')

    def clean_column_names(self, df):
        col_names = df.columns
        col_names = col_names.str.strip()            # Remove white spaces in front and at end of column names
        col_names = col_names.str.lower()            # Make all columns lower case
        col_names = col_names.str.replace(' ', '_')  # Make underscores from spaces
        col_names = col_names.str.replace('(', '')   # Remove ( 
        col_names = col_names.str.replace(')', '')   # Remove )
        df.columns = col_names
        return df            