def initialize_bucket(name_project, name_bucket):
    """Initialize the Google Cloud Bucket."""
    fs = gcsfs.GCSFileSystem(project=name_project)
    gcs = storage.Client()
    bucket = gcs.get_bucket(name_bucket)
    return bucket

def upload_blob(name_bucket, name_file_source, name_blob_destination):
    """Uploads a file to the Google Cloud Bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(name_bucket)
    blob = bucket.blob(name_blob_destination)

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

