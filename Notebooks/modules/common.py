import sys, os
import gcsfs
from google.cloud import storage

class GC_Data_Processing(object):

    def __init__(self, name_project, name_bucket):
        """Returns a new Google Cloud Bucket object."""
        cs = storage.Client()
        self.gc_fs = gcsfs.GCSFileSystem(project=name_project)
        self.gc_bucket = cs.get_bucket(name_bucket)

    def get_gc_bucket(self):
        return self.gc_bucket

    def get_gc_fs(self):
        return self.gc_fs

    def upload_blob(self, name_file_source, name_blob_destination):
        """Uploads a file to the Google Cloud Bucket."""
        blob = self.bucket.blob(name_blob_destination)

        blob.upload_from_filename(name_file_source)

        print('File {} uploaded to {}.'.format(
            name_file_source,
            name_blob_destination))

    def save_df_locally(df, dir_prefix, name_dataset, as_json= False):
        """ Saves df as json or csv locally on server """
        if as_json:        
            file_path = dir_prefix + '/dataset_' + name_dataset + '.json'
            df.to_json(file_path)
        else:
            file_path =  dir_prefix + '/dataset_' + name_dataset + '.csv'
            df.to_csv(file_path)
