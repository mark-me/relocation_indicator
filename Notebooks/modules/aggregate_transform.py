import pandas as pd
import numpy as np
import gcsfs
from google.cloud import storage
from common import *

class Aggregate_Transform(GC_Data_Processing):

    def __init__(self, name_project, name_bucket, dir_input_data, dir_output_data):
        """Returns a new Cleaner Merger object."""
        GC_Data_Processing.__init__(self, name_project, name_bucket)
        self.dir_input_data = dir_input_data
        self.dir_output_data = dir_output_data
        
    def get_dtype_clean_merge_data(self):
        # Setting up dictionary of column types
        dtype={'id_company'  :np.float64,
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

    def get_parse_dates_list_clean_merge_data(self):
        # Setting up dictionary of column types
        parse_dates = ['date_established' , 'date_financial_calamity_started',
            'date_financial_calamity_stopped', 'date_month', 'date_relocation_last']
        return parse_dates

    def get_clean_merged_csv(year, dir_prefix = ''):
        """ Reads a whole year of data from the already merged files """
        df_clean_merge = pd.DataFrame()    

        dtype = self.get_dtype_clean_merge_data()
        parse_dates = self.get_parse_dates_list_clean_merge_data()
        
        print('Starting with year: ', year)
        print(dir_prefix)
        blob_list = list(bucket.list_blobs(prefix=dir_prefix))    
        for blob in blob_list:  
            print("blob", blob.name)
            if year in blob.name:
                print('Processing file: ', blob.name)
                with fs.open('graydon-data/' + blob.name) as f:
                    full_year_df = pd.read_csv(f, sep=',', index_col=0, dtype=dtype, parse_dates=parse_dates 
                                            )   
            print('The number of rows so far is: ', full_year_df.shape[0])
        return full_year_df