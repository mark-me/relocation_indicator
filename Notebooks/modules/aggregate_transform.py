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
        self.key_aggregation = ['id_branch', 'id_company']
            def get_dtype_clean_merge_data(self):
        # Setting up dictionary of column types
        self.dtype_clean_merge = {'id_company':np.float64,  'id_branch':np.int64, 'is_discontinued':bool,
                                  'code_discontinuation':np.float64, 'code_financial_calamity':object,
                                  'financial_calamity_outcome':np.float64, 'code_legal_form':np.float64,
                                  'qty_employees':np.float64, 'year_qty_employees':np.float64,
                                  'id_company_creditproxy':object, 'score_payment_assessment':np.float64,
                                  'amt_revenue':np.float64, 'year_revenue':np.float64,
                                  'amt_operating_result':np.float64, 'year_operating_result':object, 
                                  'amt_consolidated_revenue':np.float64, 'year_consolidated_revenue':object,
                                  'amt_consolidated_operating_result':np.float64, 'year_consolidated_operating_result':object,
                                  'qty_issued_credit_reports':np.float64, 'perc_credit_limit_adjustment':object,
                                  'color_credit_status':object, 'rat_pd':object, 'score_pd': np.float64,
                                  'has_increased_risk':bool, 'is_sole_proprietor':bool, 'code_sbi_2':np.float64,
                                  'qty_address_mutations_total':np.float64, 'qty_address_mutations_month':np.float64,
                                  'has_relocated':bool, 'qty_started_names':np.float64, 'qty_stopped_names':np.float64,
                                  'total_changeof_board_members_':np.float64 }
        self.parse_dates_clean_merge = ['date_established', 'date_financial_calamity_started', 
                                        'date_financial_calamity_stopped', 'date_month', 'date_relocation_last']

    def get_merged_data(self, date_dataset):
        """Reads a whole year of data from the already merged files"""
        file_name = "cleaned_merged_" + date_dataset.strftime('%Y-%m-%d') + ".csv"
        df_clean_merge = self.get_df_from_bucket(data_out + "/" + file_output, 
                                                 dtype=self.dtype_clean_merge, 
                                                 parse_dates=self.parse_dates_clean_merge)
        return df_clean_merge

    def group_code_sbi(df):
        """ """
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
            df = df.merge(df.groupby(self.key_aggregation)[column_pair[0]] 
                          .agg('sum')             
                          .rename(column_pair[1])    
                          .to_frame()       
                          .reset_index())
        return df

    def calculate_mean(self, df):
        """The mean value."""
        list_column_pairs = [['amt_consolidated_operating_result', 'mean_amt_consolidated_operating_result'],
                             ['amt_consolidated_revenue', 'mean_amt_consolidated_revenue'],
                             ['amt_operating_result', 'mean_amt_operating_result'],
                             ['qty_issued_credit_reports', 'mean_qty_issued_credit_reports'],
                             ['score_payment_assessment', 'mean_score_payment_assessment'],
                             ['amt_revenue', 'mean_amt_revenue'],
                             ['score_pd', 'mean_score_pd']]
        for column_pair in list_column_pairs:
            df = df.merge(df.groupby(self.key_aggregation)[column_pair[0]] 
                          .agg('mean')             
                          .rename(column_pair[1])    
                          .to_frame()       
                          .reset_index()) 
        return df    

    def calculate_variance(self, df):
        """The variance of columns."""
        list_column_pairs = [['qty_employees', 'variance_qty_employees'],
                             ['qty_issued_credit_reports', 'variance_score_payment_assessment'],
                             ['score_pd', 'variance_score_pd']]
        for column_pair in list_column_pairs:
            df = df.merge(df.groupby(self.key_aggregation)[column_pair[0]] 
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
        temp_df = temp_df.groupby(self.key_aggregation).agg(['first', 'last'])

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
        """Calculate the number of years passed for date or year columns"""
        df['max_date_month'] = df.groupby(self.key_aggregation).date_month.transform('max')
        # Company age    
        df['temp_date_established_year'] = df.date_established.apply(lambda x: x.year)
        df['company_age'] = df['max_date_month_year'] - df.temp_date_established_year 
        df = df.drop(labels =['temp_date_established_year'], axis= 1)
        # Years in current location
        df['max_date_relocation_last'] = df.groupby(self.key_aggregation).date_relocation_last.transform('max')
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
        column_triples = [['amt_operating_result', 'amt_consolidated_operating_result', 'ratio_operating_result_consolidated_operating_result'],
                          ['amt_revenue', 'amt_consolidated_revenue', 'ratio_revenue_consolidated_revenue']]
        for col in column_triples:
            subset_columns =  self.key_aggregation + col[0:2]   # Create subset on which to perform the calculation
            temp_df = df.loc[:, subset_columns]
            temp_df = df.groupby(self.key_aggregation)
            temp_df = temp_df.agg({col[0]: 'sum', col[1]: 'sum'})
            temp_df[col[2]] = np.divide(temp_df[col[0]], temp_df[col[1]]) # Calculate ratios
            temp_df = temp_df.reset_index()                               # Clean up intermediate stuff
            temp_df = temp_df.drop(axis=1, columns=[col[0:2]])
            df = df.merge(temp_df, how='left', on=self.key_aggregation)  # Add back to data
        return df

    def set_true_if_any_true(self, df):
        """Set's the aggregated value to True when one of the underlying values is True""" 
        # List of 'based on'-'write to' column pairs to read-write values from-to
        list_column_pairs = [['is_discontinued', 'is_discontinued_any'],
                             ['code_financial_calamity', 'has_financial_calamity'],
                             ['has_relocated', 'has_relocated_next_year']]
        for column_pair in list_column_pairs:
            df = df.merge(df.groupby(self.key_aggregation)[column_pair[0]] 
                          .any()            
                          .rename(column_pair[1])    
                          .to_frame()       
                          .reset_index())
        return df 

    def count_dummies(self, df):
        """Creates new columns based on the frequency of the underlying values."""
        cols_count_values = [['color_credit_status', [{"G": "qty_green_flags", "O": "qty_orange_flags","R": "qty_red_flags"}]],
                             ['rat_pd', [None]],
                             ['code_SBI_2_group', [{"1": "SBI_group_1", "2": "SBI_group_2"}]], 
                             ['code_legal_form_group', [{"1": "code_legal_form_group_1", "2": "code_legal_form_group_2"}]]]

        df_dummy_count = df.groupby(self.key_aggregation)[cols_count_values[0]].value_counts()
        df_dummy_count = df_dummy_count.unstack(fill_value = 0)
        df_dummy_count = df_dummy_count.reset_index() 
        if cols_count_values[1][0] is not None:
            df_dummy_count = df_dummy_count.rename(columns=cols_count_values[1][0])

    def drop_superfluous_columns(self, df):
        """Dropping columns that aren't used after aggregation."""
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
        df = df.groupby(self.key_aggregation).first()
        df = df.reset_index()
        df = df.drop(axis=1, columns='index')
        return df 

    def impute_na_inf(self, df):

        cols_missing_to_false = ['has_financial_calamity', 'is_discontinued_any', 'SBI_has_changed', 'code_legal_form_has_changed']
        df[cols_missing_to_false] = df[cols_missing_to_false].fillna(value=False)

        cols_missing_to_zero = ['mean_qty_issued_credit_reports', 'qty_green_flags', 'qty_orange_flags', 
                                'qty_red_flags', 'AAA', 'AA', 'A', 'BBB', 'B' , 'CCC', 'CC', 'C', 'D', 
                                'NR', 'qty_address_mutations_year', 'qty_started_names_year', 
                                'qty_stopped_names_year', 'qty_board_changes_year', 'code_legal_form_group_1', 
                                'code_legal_form_group_2', 'SBI_group_1', 'SBI_group_2']
        df[cols_missing_to_zero] = df[cols_missing_to_zero].fillna(value=0)
        # Replacing infinite values with none
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

    def aggregate_transform_file(self, date_dataset):
        print("Reading merged data for", date_dataset.strftime('%Y-%m-%d'))
        df = self.get_merged_data(data_dataset)
        df = aggregate_transform_df(df)
        return(df)

    def aggregate_transform_df(self, df):
        print("Read", df.shape[0], "rows with", df.shape[1], "columns.")
        df = self.aggregate(df)
        print("writing aggregated data for", date_dataset.strftime('%Y-%m-%d'), 
              "for", df.shape[0], "rows with", df.shape[1], "columns.")
        file_output = "cleaned_merged_" + date_dataset.strftime('%Y-%m-%d') + ".csv"
        self.save_df_locally(df, self.dir_output_data, file_output)
        print("Copying file to bucket in", self.dir_output_data)
        self.local_file_to_bucket(file_source = "data_out/" + file_output,
                                  dir_bucket = self.dir_output_data)
        return(df)
