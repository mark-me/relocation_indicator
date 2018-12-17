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
        self.key = ['id_branch', 'id_company', 'date_month']
        self.key_aggregattion = ['id_branch', 'id_company']

    def get_merged_data(self, data_dataset):
        file_input = "cleaned_merged_" + date_dataset.strftime('%Y-%m-%d') + ".csv"
        parse_dates = get_parse_dates_list_clean_merge_data()
        dtype_input = self.get_dtype_clean_merge_data()
        df = self.get_df_from_bucket(file_input, dtype = dtype_input, parse_dates = parse_dates)
        return df

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
               'total_changeof_board_members_' :np.float64 }
        return dtype

    def get_parse_dates_list_clean_merge_data(self):
        # Setting up dictionary of column types
        parse_dates = ['date_established' , 
                       'date_financial_calamity_started',
                       'date_financial_calamity_stopped',
                       'date_month', 
                       'date_relocation_last']
        return parse_dates

    def get_clean_merged_csv(self, date_dataset):
        """ Reads a whole year of data from the already merged files """
        df_clean_merge = pd.DataFrame()   
        dtype = self.get_dtype_clean_merge_data()
        parse_dates = self.get_parse_dates_list_clean_merge_data()
        file_name = "cleaned_merged_" + date_dataset.strftime('%Y-%m-%d') + ".csv"
        df_clean_merge = self.get_df_from_bucket(data_out + "/" + file_output, 
                                                 dtype=dtype, parse_dates=parse_dates)
        return df_clean_merge

    def group_code_sbi(df):
        code_SBI_2_group1 = [1,19,35,51,53,59,61,62,63,69,72,73,74,78,79,80,82,85,86,87,88,90,93,94]
        df['code_SBI_2_group'] = np.where(df['code_sbi_2'].isin(code_SBI_2_group1), "1", "2")
        df = df.drop(axis=1, labels='code_sbi_2', inplace=False)
        return df
    
    def group_code_legal_form(df):
        code_legal_form_group = [1,4,6,7,8,9,15,17,18]
        df['code_legal_form_group'] = np.where(df['code_legal_form'].isin(code_legal_form_group), "1", "2")
        df = df.drop(axis=1, labels='code_legal_form', inplace=False)
        return df    

    def calculate_sum(self, df):
        # List of 'based on'-'write to' column pairs to read-write values from-to
        list_column_pairs = [['qty_address_mutations_month', 'qty_address_mutations_year'],
                             ['qty_started_names', 'qty_started_names_year'],
                             ['qty_stopped_names', 'qty_stopped_names_year'],
                             ['total_changeof_board_members_', 'qty_board_changes_year']]
        for column_pair in list_column_pairs:
            df = df.merge(df.groupby(self.key_aggregattion)[column_pair[0]] 
                          .agg('sum')             
                          .rename(column_pair[1])    
                          .to_frame()       
                          .reset_index())
        return df

    def calculate_mean(self, df):
        list_column_pairs = [['amt_consolidated_operating_result', 'mean_amt_consolidated_operating_result'],
                             ['amt_consolidated_revenue', 'mean_amt_consolidated_revenue'],
                             ['amt_operating_result', 'mean_amt_operating_result'],
                             ['qty_issued_credit_reports', 'mean_qty_issued_credit_reports'],
                             ['score_payment_assessment', 'mean_score_payment_assessment'],
                             ['amt_revenue', 'mean_amt_revenue'],
                             ['score_pd', 'mean_score_pd']]
        for column_pair in list_column_pairs:
            df = df.merge(df.groupby(self.key_aggregattion)[column_pair[0]] 
                          .agg('mean')             
                          .rename(column_pair[1])    
                          .to_frame()       
                          .reset_index()) 
        return df    

    def calculate_variance(self, df):
        list_column_pairs = [['qty_employees', 'variance_qty_employees'],
                             ['qty_issued_credit_reports', 'variance_score_payment_assessment'],
                             ['score_pd', 'variance_score_pd']]
        for column_pair in list_column_pairs:
            df = df.merge(df.groupby(self.key_aggregattion)[column_pair[0]] 
                          .agg('var')             
                          .rename(column_pair[1])    
                          .to_frame()       
                          .reset_index()) 
        return df       

    def calculate_deltas(self, df):
        """Calculate difference in value between first and last date in the dataset."""
        # Sorting dataset by date_month and getting first and last values per branch
        list_column_pairs=[['qty_employees', 'delta_qty_employees'], 
                           ['qty_issued_credit_reports', 'delta_qty_issued_credit_reports'], 
                           ['score_payment_assessment', 'delta_score_payment_assessment'], 
                           ['score_pd', 'delta_score_pd'],
                           ['code_legal_form', 'code_legal_form_has_changed'], 
                           ['code_SBI_2_group', 'SBI_has_changed']]
        subset_columns = self.key
        subset_columns.extend([item[0] for item in list_column_pairs])
        temp_df = df.reset_index().loc[:, subset_columns].sort_values(self.key)
        temp_df = temp_df.groupby(self.key_aggregattion).agg(['first', 'last'])

        # Calculating delta's
        for column_pair in list_column_pairs:
            temp_df[column_pair[1]] = temp_df[column_pair[0]]['last'] - temp_df[column_pair[0]]['first']    
        
        # Ungroup and deduplicate columns  
        temp_df.columns = temp_df.columns.droplevel(1)
        temp_df = temp_df.loc[:,~temp_df.columns.duplicated()]
        temp_df = temp_df.drop(axis=1, columns=col_list)

        # Add delta's to original data-frame
        df = df.merge(temp_df, how='left', on=['date_month', 'id_company', 'id_branch']) 
        return df 

    def calculate_age(self, df):
       
        df['max_date_month'] = df.groupby(self.key_aggregattion).date_month.transform('max')
        # Company age    
        df['temp_date_established_year'] = df.date_established.apply(lambda x: x.year)
        df['company_age'] = df['max_date_month_year'] - df.temp_date_established_year 
        df = df.drop(labels =['temp_date_established_year'], axis= 1)
        # Years in current location
        df['max_date_relocation_last'] = df.groupby(self.key_aggregattion).date_relocation_last.transform('max')
        df['temp_max_date_relocation_last_year'] = df.max_date_relocation_last.apply(lambda x: x.year)
        df['years_in_current_location'] = df['max_date_month_year'] - df.temp_max_date_relocation_last_year 
        df = df.drop(labels =['temp_max_date_relocation_last_year', 'max_date_relocation_last'], axis= 1)
        # Calculate age based of 'year' columns
        list_pairs_year = [['year_consolidated_operating_result', 'years_since_last_amt_consolidated_operating_result'],
                           ['year_consolidated_revenue', 'years_since_last_amt_consolidated_revenue'],
                           ['year_operating_result', 'years_since_last_amt_operating_result'],
                           ['year_qty_employees', 'years_since_last_qty_employees'],
                           ['year_revenue', 'years_since_last_amt_revenue']]
        df['max_date_month_year'] = df['max_date_month'].apply(lambda x: x.year)

        for column_pair in list_pairs_year:
            mask = (df[column_pair[0]].astype(float) > 0)
            df_valid = df[mask]
            df[column_pair[1]] = np.nan
            df.loc[mask, column_pair[1]] = (df['max_date_month_year'] - df_valid[column_pairp[0]].astype(float))  
 
        # Drop intermediate calculating columns
        df = df.drop(labels =['max_date_month', 'max_date_month_year'], axis= 1)
        return df

    def calculate_ratios(self, df):
        col_list = ['amt_operating_result', 'amt_consolidated_operating_result',
                    'amt_revenue', 'amt_consolidated_revenue']
        for col in col_list:
            subset_columns = ['id_branch']
            subset_columns.extend(col)
            temp_df = df.loc[:, subset_columns]
            if col == 'amt_operating_result':
                temp_df = df.groupby(self.key_aggregattion)
                temp_df = temp_df.agg({'amt_operating_result': 'sum', 'amt_consolidated_operating_result': 'sum'}).rename(
        columns={'amt_operating_result': 'sum_amt_operating_result', 
                 'amt_consolidated_operating_result': 'sum_amt_consolidated_operating_result'})
                temp_df['ratio_operating_result_consolidated_operating_result'] = np.divide(
                    temp_df['sum_amt_operating_result'], temp_df['sum_amt_consolidated_operating_result'])
                temp_df = temp_df.reset_index()
                temp_df = temp_df.drop(axis=1, columns=['sum_amt_consolidated_operating_result', 
                                                        'sum_amt_operating_result'])
                df = df.merge(temp_df, how='left', on= ['id_branch', 'id_company'])  
            elif col == 'amt_revenue':
                temp_df = df.groupby(self.key_aggregattion])
                temp_df = temp_df.agg({'amt_revenue': 'sum', 
                    'amt_consolidated_revenue': 'sum'}).rename(columns={'amt_revenue': 'sum_amt_revenue', 
                                                                        'amt_consolidated_revenue': 'sum_amt_consolidated_revenue'})
                temp_df['ratio_revenue_consolidated_revenue'] = np.divide(temp_df['sum_amt_revenue'],
                                                                         temp_df['sum_amt_consolidated_revenue'])
                temp_df = temp_df.reset_index()
                temp_df = temp_df.drop(axis=1, columns=['sum_amt_revenue', 'sum_amt_consolidated_revenue'])
                df = df.merge(temp_df, how='left',  on= ['id_branch', 'id_company'])  
        return df

    def set_true_if_any_true(self, df):
        # List of 'based on'-'write to' column pairs to read-write values from-to
        list_column_pairs = [['is_discontinued', 'is_discontinued_any'],
                             ['code_financial_calamity', 'has_financial_calamity'],
                             ['has_relocated', 'has_relocated_next_year']]
        for column_pair in list_column_pairs:
            df = df.merge(df.groupby(self.key_aggregattion)[column_pair[0]] 
                          .any()            
                          .rename(column_pair[1])    
                          .to_frame()       
                          .reset_index())
        return df 

    def count_dummies(self, df):
        col_list = ['color_credit_status','rat_pd', 'code_legal_form_group', 'code_SBI_2_group']
        df = df.reset_index()
        subset_columns = ['id_branch']
        subset_columns.extend(col_list)
        df['unique_id'] =  df['id_branch'].astype(str) + '_' + df['id_company'].astype(str)
        for col in col_list:
            temp_df = df.loc[:, subset_columns]
            if col == 'color_credit_status':
                temp_df = pd.crosstab(df['unique_id'], df['color_credit_status']).reset_index().rename_axis(None,
                                                                                                            axis=1).rename(
                    columns={"G": "qty_green_flags", "O": "qty_orange_flags","R": "qty_red_flags"})
            elif col == 'rat_pd':
                temp_df = pd.crosstab(df['unique_id'], df['rat_pd']).reset_index().rename_axis(None, axis=1)
            elif col == 'code_SBI_2_group':
                temp_df = pd.crosstab(df['unique_id'], df['code_SBI_2_group']).reset_index().rename_axis(None,
                                                                                                            axis=1).rename(
                    columns={"1": "SBI_group_1", "2": "SBI_group_2"})
            elif col == 'code_legal_form_group':
                temp_df = pd.crosstab(df['unique_id'], df['code_legal_form_group']).reset_index().rename_axis(None,
                                                                                                            axis=1).rename(
                    columns={"1": "code_legal_form_group_1", "2": "code_legal_form_group_2"})
            df = df.merge(temp_df, how='left', on= ['unique_id']) 
        return df

    def drop_superfluous_columns(self, df):
        columns_superfluous = ['date_established', 'year_consolidated_operating_result', 'year_consolidated_revenue', 
                               'year_operating_result', 'year_qty_employees', 'year_revenue',
                               'is_discontinued', 'code_financial_calamity', 'amt_consolidated_operating_result',  
                               'amt_consolidated_revenue', 'amt_operating_result','amt_revenue','qty_employees', 
                               'qty_issued_credit_reports', 'score_payment_assessment', 'score_pd', 
                               'color_credit_status', 'rat_pd', 'qty_address_mutations_month','qty_started_names',
                               'qty_stopped_names', 'total_changeof_board_members_', 'is_sole_proprietor',
                               'code_discontinuation','date_financial_calamity_started', 'date_financial_calamity_stopped', 
                               'id_company_creditproxy', 'financial_calamity_outcome', 'has_increased_risk' , 
                               'perc_credit_limit_adjustment', 'date_start', 'from_date_start', 
                               'qty_address_mutations_total', 'code_legal_form_group', 'code_SBI_2_group', 
                               'date_relocation_last', 'date_relocation_penultimate']
        df = df.drop(axis=1, labels=columns_superfluous, inplace=False)  
        return df

    def deduplicate_rows(self, df):
        df = df.groupby(self.key_aggregattion).first()
        df = df.reset_index()
        df = df.drop(axis=1, columns='index')
        return df 

    def impute_na_inf(self, df):
        df[['has_financial_calamity', 'is_discontinued_any', 
            'SBI_has_changed', 'code_legal_form_has_changed']] = df[['has_financial_calamity', 'is_discontinued_any',
                                                                     'code_legal_form_has_changed', 'SBI_has_changed']].fillna(value=False)
        columns_to_zero = ['mean_qty_issued_credit_reports', 'qty_green_flags', 'qty_orange_flags', 
                           'qty_red_flags', 'AAA', 'AA', 'A', 'BBB', 'B' , 'CCC', 'CC', 'C', 'D', 
                           'NR', 'qty_address_mutations_year', 'qty_started_names_year', 
                           'qty_stopped_names_year', 'qty_board_changes_year', 'code_legal_form_group_1', 
                           'code_legal_form_group_2', 'SBI_group_1', 'SBI_group_2']
        df[columns_to_zero] = df[columns_to_zero].fillna(value=0)
        df = df.replace([np.inf, -np.inf], np.nan)
        return df 

    def aggregate(self, df):
        """Aggregating dataframe into one year. Main function that calls them all"""
        print('Creating SBI groups')
        df = self.group_code_sbi(df)
        print('Creating code legal form groups')
        df = self.group_code_legal_form(df)
        print('Calculating sum of columns')
        df = self.calculate_sum(df)
        print('Calculating mean of columns')
        df = self.calculate_mean(df)
        print('Calculating ages')
        df = self.calculate_age_based_on_date(df)
        print('Creating counts from dummies')
        df = self.count_dummies(df)
        print('Calculating variance of columns')
        df = self.calculate_variance(df) 
        print('Calculating if any true')
        df = self.set_true_if_any_true(df)
        print('Calculating ratios')
        df = self.calculate_ratios(df)        
        print('Calculating deltas')
        df = self.calculate_deltas(df)
        print('Dropping old columns')
        df = self.drop_superfluous_columns(df)
        print('Deduplicating rows of original dataframe')
        df = self.deduplicate_rows(df)
        print('Setting NaN and Inf values')
        df = self.impute_na_inf(df)
        return df    

    def aggregate_transform_data(self, date_dataset):
        print("Reading merged data for", date_dataset.strftime('%Y-%m-%d'))
        df = self.get_merged_data(data_dataset)
        print("Read", df.shape[0], "rows with", df.shape[1], "columns.")
        df = self.aggregate(df)
        print("writing aggregated data for", date_dataset.strftime('%Y-%m-%d'), 
              "for", df.shape[0], "rows with", df.shape[1], "columns.")
        file_output = "cleaned_merged_" + date_dataset.strftime('%Y-%m-%d') + ".csv"
        self.save_df_locally(df, self.dir_output_data, file_output)
        print("Copying file to bucket in", self.dir_output_data)
        self.local_file_to_bucket(file_source = "data_out/" + file_output,
                                  dir_bucket = self.dir_output_data)
