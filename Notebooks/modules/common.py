import gcsfs
from google.cloud import storage

def save_df_locally(df, dir_prefix, name_dataset, as_json= False):
    """ Saves df as json or csv locally on server """
    if as_json:        
        file_path = dir_prefix + '/dataset_' + name_dataset + '.json'
        df.to_json(file_path)
    else:
        file_path =  dir_prefix + '/dataset_' + name_dataset + '.csv'
        df.to_csv(file_path)


class GCS_Bucket(object):

    def __init__(self, name_project, name_bucket):
        """Returns a new Google Cloud Bucket object."""
        cs = storage.Client()
        self.fs = gcsfs.GCSFileSystem(project=name_project)
        self.bucket = cs.get_bucket(name_bucket)

    def get_bucket(self):
        return self.bucket

    def get_fs(self):
        return self.fs

    def upload_blob(self, name_file_source, name_blob_destination):
        """Uploads a file to the Google Cloud Bucket."""
        blob = self.bucket.blob(name_blob_destination)

        blob.upload_from_filename(name_file_source)

        print('File {} uploaded to {}.'.format(
            name_file_source,
            name_blob_destination))
