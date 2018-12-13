import pandas as pd
import gcsfs
from google.cloud import storage
from pandas import DataFrame
from IPython.display import HTML
from google.cloud.storage import Blob
import datalab.storage as gcs_datalab
from datetime import date
import numpy as np


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

# Pre-aggregation
def create_dict_types_original_data():
    # Setting up dictionary of column types
    dtype={ 'id_company'  :np.float64,
        'id_branch'    :np.int64,
        'is_discontinued':bool,
        'code_discontinuation': np.float64,
        'code_financial_calamity':object,
        'financial_calamity_outcome'   : np.float64,
        'code_legal_form' : np.float64,
        'qty_employees' :np.float64,
        'year_qty_employees' :np.float64,
        'id_company_creditproxy':object,
        'score_payment_assessment'    : np.float64,
        'amt_revenue'  : np.float64,
        'year_revenue'  : np.float64,
        'amt_operating_result'   : np.float64,
        'year_operating_result'    :object,
        'amt_consolidated_revenue'   : np.float64,
        'year_consolidated_revenue'   :object,
        'amt_consolidated_operating_result'     : np.float64,
        'year_consolidated_operating_result'   :object,
        'qty_issued_credit_reports' : np.float64,
        'perc_credit_limit_adjustment' :object,
        'color_credit_status'  :object,
        'rat_pd'              :object,
        'score_pd'            : np.float64,
        'has_increased_risk'  :bool,
        'is_sole_proprietor'   :bool,
        'code_sbi_2'         : np.float64,
        'qty_address_mutations_total'  :np.float64,
        'qty_address_mutations_month'   :np.float64,
        'has_relocated':bool,
        'qty_started_names': np.float64,
        'qty_stopped_names': np.float64,
        'total_changeof_board_members_' :np.float64
    }
	return dtype

def create_parse_dates_list_original_data():
    # Setting up dictionary of column types
    parse_dates = ['date_established' , 'date_financial_calamity_started',
        'date_financial_calamity_stopped', 'date_start', 'from_date_start', 'date_month' ]
    return parse_dates
	
# Post-aggregation
def create_dict_types_aggregated_data():
    # Setting up dictionary of column types for the aggregated dataset
	dtype={ 
	    'id_company'  :np.float64,
        'id_branch'    :np.int64, 
        'code_sbi_2'         : np.float64, 
        'has_relocated':bool,
        'has_relocated_next_year ' : bool,
		'has_name_change' : bool,
        'qty_address_mutations_total' :np.float64,
        'ratio_operating_result_consolidated_operating_result': np.float64,
        'ratio_revenue_consolidated_revenue': np.float64,
        'qty_green_flags'   :np.float64,
        'qty_orange_flags'   :np.float64,
        'qty_red_flags'   :np.float64,
        'A'   :np.float64,
        'AA'   :np.float64,
        'AAA'   :np.float64,
        'B'   :np.float64,
        'BB'   :np.float64,
        'BBB'   :np.float64,
        'C'   :np.float64,
        'CC'   :np.float64,
        'CCC'   :np.float64,
        'D'   :np.float64,
        'NR'   :np.float64,
		'code_legal_form_group_1':  int64,
        'code_legal_form_group_2':  int64,
		'SBI_group_1':  int64,
        'SBI_group_2':  int64,
        'company_age'   :np.float64,
        'years_since_last_amt_consolidated_operating_result'   :np.float64,
        'years_since_last_amt_consolidated_revenue'   :np.float64,
        'years_since_last_amt_operating_result'   :np.float64,
        'years_since_last_qty_employees'   :np.float64,
        'years_since_last_amt_revenue'   :np.float64,
        'delta_qty_employees'   :np.float64,
        'delta_qty_issued_credit_reports'   :np.float64,
        'delta_score_payment_assessment'   :np.float64,
		'SBI_has_changed' : bool,
		'unique_id' : object,
        'code_legal_form_has_changed ' : bool,
        'is_discontinued_any ' : bool,
        'has_financial_calamity ' : bool,
        'mean_amt_consolidated_operating_result'   :np.float64,
        'mean_amt_consolidated_revenue'   :np.float64,
        'mean_amt_operating_result'   :np.float64,
        'mean_amt_revenue'   :np.float64,
        'mean_qty_employees'   :np.float64,
        'mean_qty_issued_credit_reports'   :np.float64,
        'mean_score_payment_assessment'   :np.float64,
        'mean_score_pd'   :np.float64,
        'qty_address_mutations_year'   :np.float64,
        'qty_started_names_year'   :np.float64,
        'qty_stopped_names_year'   :np.float64,
        'qty_board_changes_year'   :np.float64,
		'variance_qty_employees'   :np.float64,
        'variance_qty_issued_credit_reports'   :np.float64,
        'variance_score_payment_assessment'   :np.float64,
        'variance_score_pd'   :np.float64
      }
	return dtype


def create_parse_dates_list_aggregated_data():
    # Setting up dictionary of column types for the aggregated dataset
    parse_dates= ['date_month', 'date_relocation_last', 'date_relocation_penultimate']
    return parse_dates
	

def read_one_year_from_bucket_merged_csv(year, dir_prefix = ''):
    """ Reads a whole year of data from the already merged files. Can be use as input for the aggregated function """
    dtype = create_dict_types_original_data()
	parse_dates = create_parse_dates_list_original_data()
    full_year_df = pd.DataFrame()
    print('Starting with year: ', year)
    print(dir_prefix)
    blob_list = list(bucket.list_blobs(prefix=dir_prefix))    
    for blob in blob_list:  
        print("blob", blob.name)
        if year in blob.name:
            print('Processing file: ', blob.name)
            with fs.open('graydon-data/' + blob.name) as f:
                full_year_df = pd.read_csv(f, sep=',', index_col=0, dtype=dtype, parse_dates=parse_dates)   
        print('The number of rows so far is: ', full_year_df.shape[0])
    return full_year_df

# Pre-aggregation
def read_all_csv_months_yearly_from_bucket_merged(years_to_read_in_list, dir_prefix = '', selected_columns = ''):
    """ Reads a whole year of data based on monthly original files and returns a monthly cleaned merged pandas Df. """
    all_years_merged_df = pd.DataFrame()
    for year in years_to_read_in_list:
        print('Starting with year: ', year)
        dir_prefix = dir_prefix + '/' + year
        blob_list = list(bucket.list_blobs(prefix=dir_prefix))    
        for blob in blob_list:  
            one_month_df = None
            if 'CSV' in blob.name:
                print('Processing file: ', blob.name)
                with fs.open('graydon-data/' + blob.name) as f:
                    one_month_df = pd.read_csv(f, sep=';', usecols= selected_columns)   
                    one_month_df = one_month_df[(one_month_df['is_sole_proprietor'] == 0) ]
                                               # & (one_month_df['is_discontinued'] == 0) 
                    one_month_df.columns = (one_month_df.columns.str.strip().str.lower(). 
                    str.replace(' ', '_').str.replace('(', '').str.replace(')', '') )
                    one_month_df = aggregate_board_members(one_month_df)
                    one_month_df = clean_data_per_year(one_month_df)
                    all_years_merged_df = all_years_merged_df.append(one_month_df)
            print('The number of rows so far is: ', all_years_merged_df.shape[0])
    return all_years_merged_df
	
def create_basetable(year_list, dir_prefix = ''):
    """ Reads a whole year of data from the already aggregated files and creates basetable """
    dtype = create_dict_types_aggregated_data()
    parse_dates = create_parse_dates_list_aggregated_data()
    basetable = pd.DataFrame()
    for year in year_list:
        full_year_df = pd.DataFrame()
        print('Starting with year: ', year)
        print(dir_prefix)
        blob_list = list(bucket.list_blobs(prefix=dir_prefix))    
        for blob in blob_list:  
            if year in blob.name:
                print('Processing file: ', blob.name)
                with fs.open('graydon-data/' + blob.name) as f:
                    full_year_df = pd.read_csv(f, sep=',', index_col=0, dtype=dtype, parse_dates=parse_dates 
                                            )   
                print('The number of rows of the year read is far is: ', full_year_df.shape[0])
        basetable = basetable.append(full_year_df)
    print('The final number of rows of the basetable created is: ', basetable.shape[0])
    return basetable


      