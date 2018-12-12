import pandas as pd
import gcsfs
from google.cloud import storage
from pandas import DataFrame
from IPython.display import HTML
from google.cloud.storage import Blob
import datalab.storage as gcs_datalab
from datetime import date
import numpy as np

# Age of dates
def calculate_age_based_on_date(df, col_list):
    df['max_date_month'] = df.groupby(['id_branch', 'id_company']).date_month.transform('max')
    df['max_date_month_year'] = df['max_date_month'].apply(lambda x: x.year)
    for col in col_list:
        if col == 'date_established':
            df['temp_date_established_year'] = df.date_established.apply(lambda x: x.year)
            df['company_age'] = df['max_date_month_year'] - df.temp_date_established_year 
            df = df.drop(labels =['temp_date_established_year'], axis= 1)
        elif col == 'year_consolidated_operating_result':            
            mask = (df['year_consolidated_operating_result'].astype(float) > 0)
            df_valid = df[mask]
            df['years_since_last_amt_consolidated_operating_result'] = np.nan
            df.loc[mask, 'years_since_last_amt_consolidated_operating_result'] = (df['max_date_month_year'] - 
                                df_valid.year_consolidated_operating_result.astype(float))  
        elif col == 'year_consolidated_revenue':
            mask = (df['year_consolidated_revenue'].astype(float) > 0)
            df_valid = df[mask]
            df['years_since_last_amt_consolidated_revenue'] = np.nan
            df.loc[mask, 'years_since_last_amt_consolidated_revenue'] = (df['max_date_month_year'] - 
                                df_valid.year_consolidated_revenue.astype(float))    
        elif col == 'year_operating_result':
            mask = (df['year_operating_result'].astype(float) > 0)
            df_valid = df[mask]
            df['years_since_last_amt_operating_result'] = np.nan
            df.loc[mask, 'years_since_last_amt_operating_result'] = (df['max_date_month_year'] - 
                                df_valid.year_operating_result.astype(float))    
        elif col == 'year_qty_employees':
            mask = (df['year_qty_employees'].astype(float) > 0)
            df_valid = df[mask]
            df['years_since_last_qty_employees'] = np.nan
            df.loc[mask, 'years_since_last_qty_employees'] = (df['max_date_month_year'] - 
                                df_valid.year_qty_employees.astype(float))  
        elif col == 'year_revenue':
            mask = (df['year_revenue'].astype(float) > 0)
            df_valid = df[mask]
            df['years_since_last_amt_revenue'] = np.nan
            df.loc[mask, 'years_since_last_amt_revenue'] = (df['max_date_month_year'] - 
                                df_valid.year_revenue.astype(float)) 
    df = df.drop(labels =['max_date_month', 'max_date_month_year'], axis= 1)
    return df

# Delta
def calculate_delta_of_column(df, col_list):
    subset_columns = ['date_month', 'id_company', 'id_branch']
    subset_columns.extend(col_list)
    temp_df = df.reset_index().loc[:, subset_columns].sort_values(['id_company','id_branch', 'date_month'])
    temp_df = temp_df.groupby(['id_branch', 'id_company']).agg(['first', 'last'])
    for col in col_list:
        if col == 'qty_employees':
            temp_df['delta_qty_employees'] = temp_df['qty_employees']['last'] - temp_df['qty_employees']['first']    
        elif col == 'qty_issued_credit_reports':
            temp_df['delta_qty_issued_credit_reports'] = (temp_df['qty_issued_credit_reports']['last'] - 
                                                          temp_df['qty_issued_credit_reports']['first'] )
        elif col == 'score_payment_assessment':
            temp_df['delta_score_payment_assessment'] = (temp_df['score_payment_assessment']['last'] - 
                                                          temp_df['score_payment_assessment']['first'] )
        elif col == 'score_pd':
            temp_df['delta_score_pd'] = (temp_df['score_pd']['last'] - 
                                                          temp_df['score_pd']['first'] )
        elif col == 'code_legal_form':
            temp_df['code_legal_form_has_changed'] = (temp_df['code_legal_form']['last'] !=
                                                          temp_df['code_legal_form']['first'] )
        elif col == 'code_SBI_2_group':
            temp_df['SBI_has_changed'] = (temp_df['code_SBI_2_group']['last'] !=
                                                          temp_df['code_SBI_2_group']['first'] )   
    temp_df.columns = temp_df.columns.droplevel(1)
    temp_df = temp_df.loc[:,~temp_df.columns.duplicated()]
    temp_df = temp_df.drop(axis=1, columns=col_list)        
    df = df.merge(temp_df, how='left', on=['date_month', 'id_company', 'id_branch']) 
    return df

# If any true then true
def calculate_if_any_true(df, col_list):
    for col in col_list:
        if col == 'is_discontinued': 
            df = df.merge(df.groupby(['id_branch', 'id_company'])['is_discontinued'] 
                        .any()              # True if any items are True
                        .rename('is_discontinued_any')    # name Series 
                        .to_frame()         # make a dataframe for merging
                        .reset_index())
        elif col == 'code_financial_calamity':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['code_financial_calamity'] 
                        .any()            
                        .rename('has_financial_calamity')   
                        .to_frame() 
                        .reset_index())
        elif col == 'has_relocated':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['has_relocated'] 
                        .any()            
                        .rename('has_relocated_next_year')   
                        .to_frame() 
                        .reset_index())
    return df

# Mean
def calculate_mean_of_column(df, col_list):
    for col in col_list:
        if col == 'amt_consolidated_operating_result':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['amt_consolidated_operating_result'] 
                        .agg('mean')             
                        .rename('mean_amt_consolidated_operating_result')    
                        .to_frame()       
                        .reset_index())
        if col == 'amt_consolidated_revenue':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['amt_consolidated_revenue'] 
                        .agg('mean')              
                        .rename('mean_amt_consolidated_revenue')    
                        .to_frame()      
                        .reset_index())
        if col == 'amt_operating_result':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['amt_operating_result'] 
                        .agg('mean')           
                        .rename('mean_amt_operating_result')    
                        .to_frame()         
                        .reset_index())
        if col == 'amt_revenue':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['amt_revenue']
                        .agg('mean')        
                        .rename('mean_amt_revenue')   
                        .to_frame()     
                        .reset_index())
        if col == 'qty_employees':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['qty_employees'] 
                        .agg('mean')          
                        .rename('mean_qty_employees')    
                        .to_frame()       
                        .reset_index())
        if col == 'qty_issued_credit_reports':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['qty_issued_credit_reports'] 
                        .agg('mean')       
                        .rename('mean_qty_issued_credit_reports')    
                        .to_frame()        
                        .reset_index())
        if col == 'score_payment_assessment':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['score_payment_assessment'] 
                        .agg('mean')       
                        .rename('mean_score_payment_assessment')    
                        .to_frame()        
                        .reset_index())
        if col == 'score_pd':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['score_pd'] 
                        .agg('mean')       
                        .rename('mean_score_pd')    
                        .to_frame()        
                        .reset_index())        
    return df

# Dummies into counts
def column_dummies_into_counts(df, col_list):
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

# Ratio
def calculate_ratio_of_column(df, col_list):
    for col in col_list:
        subset_columns = ['id_branch']
        subset_columns.extend(col)
        temp_df = df.loc[:, subset_columns]
        if col == 'amt_operating_result':
            temp_df = df.groupby(['id_branch', 'id_company'])
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
            temp_df = df.groupby(['id_branch', 'id_company'])
            temp_df = temp_df.agg({'amt_revenue': 'sum', 'amt_consolidated_revenue': 'sum'}).rename(
    columns={'amt_revenue': 'sum_amt_revenue', 
             'amt_consolidated_revenue': 'sum_amt_consolidated_revenue'})
            temp_df['ratio_revenue_consolidated_revenue'] = np.divide(temp_df['sum_amt_revenue'],
                                                                     temp_df['sum_amt_consolidated_revenue'])
            temp_df = temp_df.reset_index()
            temp_df = temp_df.drop(axis=1, columns=['sum_amt_revenue', 'sum_amt_consolidated_revenue'])
            df = df.merge(temp_df, how='left',  on= ['id_branch', 'id_company'])  
    return df

# Sum
def calculate_sum_of_column(df, col_list):
    for col in col_list:
        if col == 'qty_address_mutations_month':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['qty_address_mutations_month'] 
                        .agg('sum')             
                        .rename('qty_address_mutations_year')    
                        .to_frame()       
                        .reset_index())
        elif col == 'qty_started_names':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['qty_started_names'] 
                        .agg('sum')              
                        .rename('qty_started_names_year')    
                        .to_frame()      
                        .reset_index())
        elif col == 'qty_stopped_names':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['qty_stopped_names'] 
                        .agg('sum')           
                        .rename('qty_stopped_names_year')    
                        .to_frame()         
                        .reset_index())
        elif col == 'total_changeof_board_members_':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['total_changeof_board_members_']
                        .agg('sum')        
                        .rename('qty_board_changes_year')   
                        .to_frame()     
                        .reset_index())
    return df


# Variance
def calculate_variance_of_column(df, col_list):
    for col in col_list:
        if col == 'qty_employees':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['qty_employees'] 
                        .agg('var')             
                        .rename('variance_qty_employees')    
                        .to_frame()       
                        .reset_index())
        elif col == 'qty_issued_credit_reports':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['qty_issued_credit_reports'] 
                        .agg('var')              
                        .rename('variance_qty_issued_credit_reports')    
                        .to_frame()      
                        .reset_index())
        elif col == 'score_payment_assessment':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['score_payment_assessment'] 
                        .agg('sum')           
                        .rename('variance_score_payment_assessment')    
                        .to_frame()         
                        .reset_index())
        elif col == 'score_pd':
            df = df.merge(df.groupby(['id_branch', 'id_company'])['score_pd']
                        .agg('sum')        
                        .rename('variance_score_pd')   
                        .to_frame()     
                        .reset_index())
    return df

# Adds has_relocated_next_year to the DF	
def replace_has_relocated_with_nextyear(df, next_year, dir_prefix = ''):
    dtype={ 
            'id_branch'    :np.int64,
            'id_company'    :np.int64,
            'has_relocated':bool
    }
    full_next_year_df = pd.DataFrame()
    cols = ['id_company', 'id_branch', 'has_relocated']
    print('Starting withGra year: ', next_year)
    print(dir_prefix)
    blob_list = list(bucket.list_blobs(prefix=dir_prefix))    
    for blob in blob_list:         
        if next_year in blob.name:
            print('Processing file: ', blob.name)
            with fs.open('graydon-data/' + blob.name) as f:
                full_next_year_df = pd.read_csv(f, sep=',',  dtype=dtype, usecols= cols
                                        )   
        print('The number of rows so far is: ', full_next_year_df.shape[0])
    full_next_year_df = calculate_if_any_true(full_next_year_df, col_list = ['has_relocated'])
    full_next_year_df = full_next_year_df.drop(axis=1, columns='has_relocated')
    full_next_year_df = full_next_year_df.drop_duplicates().reset_index().drop(axis=1, columns='index')
    df = df.merge(full_next_year_df, on=['id_branch', 'id_company'], how='left', suffixes='_C')
    return df


# Creating SBI code groups
def create_sbi_groups(df):
    code_SBI_2_group1 = [1,19,35,51,53,59,61,62,63,69,72,73,74,78,79,80,82,85,86,87,88,90,93,94]
    df['code_SBI_2_group'] = np.where(df['code_sbi_2'].isin(code_SBI_2_group1), "1", "2")
    df = df.drop(axis=1, labels='code_sbi_2', inplace=False)
    return df


# Creating code legal from groups
def create_code_legal_form_groups(df):
    code_legal_form_group = [1,4,6,7,8,9,15,17,18]
    df['code_legal_form_group'] = np.where(df['code_legal_form'].isin(code_legal_form_group), "1", "2")
    df = df.drop(axis=1, labels='code_legal_form', inplace=False)
    return df


#  Dropping pre-aggregation columns
def drop_old_columns(df, col_list):
    df = df.drop(axis=1, labels=col_list, inplace=False)   
    return df

# Keeping only the first row for each branch group
def deduplicate_rows(df):
    df = one_year_df.groupby(['id_branch', 'id_company']).first()
    df = df.reset_index()
    df = df.drop(axis=1, columns='index')
    return df    

# Cleaning after aggregation
def clean_after_aggregations(df):
    df[['has_financial_calamity', 'is_discontinued_any', 'SBI_has_changed'
        ,'code_legal_form_has_changed']] = df[['has_financial_calamity', 'is_discontinued_any',
                                                'code_legal_form_has_changed', 'SBI_has_changed']].fillna(value=False)
    
    columns_to_zero = ['mean_qty_issued_credit_reports', 'qty_green_flags', 'qty_orange_flags', 
                       'qty_red_flags', 'AAA', 'AA', 'A', 'BBB', 'B' , 'CCC', 'CC', 'C', 'D', 
                       'NR', 'qty_address_mutations_year', 'qty_started_names_year', 
                       'qty_stopped_names_year', 'qty_board_changes_year', 'code_legal_form_group_1', 
                       'code_legal_form_group_2', 'SBI_group_1', 'SBI_group_2']
    
    df[columns_to_zero] = df[columns_to_zero].fillna(value=0)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df 

# Aggregating dataframe into one year. Main function that calls them all
def aggregate_full_year(year, dir_prefix = '', save_df_locally = False):
    
    print('Reading DF for year ', year) 
	df = read_one_year_from_bucket_merged_csv(year, dir_prefix)
	
    print('Creating SBI groups ')
    df = create_sbi_groups(df)
    print('Done creating SBI groups')
    
    print('Calculating delta of variables ')
    df = calculate_delta_of_column(df, col_list=['qty_employees','qty_issued_credit_reports', 
                                                        'score_payment_assessment',
                                                       'code_legal_form', 'code_SBI_2_group'])
    print('Done calculating delta of variables ') 
    
    print('Creating code legal form groups ')
    df = create_code_legal_form_groups(df)
    print('Done creating code legal form groups')
    
    print('Calculating ages of variables ')
    df = calculate_age_based_on_date(df,['date_established', 'year_consolidated_operating_result', 
                                         'year_consolidated_revenue',
                                        'year_operating_result', 'year_qty_employees', 'year_revenue'])
    print('Done calculating ages of variables ')
    
    print('Calculating ratio of columns')
    df = calculate_ratio_of_column(df, col_list=['amt_operating_result',
                                                        'amt_consolidated_operating_result',
                                                         'amt_revenue',
                                                        'amt_consolidated_revenue'])
    print('Done calculating ratio of columns')
        
    print('Making dummies into counts')
    df = column_dummies_into_counts(df, col_list=['color_credit_status','rat_pd', 'code_legal_form_group',
                                                  'code_SBI_2_group'])
    print('Done making dummies into counts')
    
 
    print('Calculating if any true ')
    df = calculate_if_any_true(df, col_list=['is_discontinued', 'code_financial_calamity'])
    print('Done calculating if any true ')
    
    print('Calculating mean of columns ')
    df = calculate_mean_of_column(df, col_list=['amt_consolidated_operating_result', 
                                                        'amt_consolidated_revenue',
                                                       'amt_operating_result','amt_revenue',
                                                       'qty_employees', 'qty_issued_credit_reports',
                                                      'score_payment_assessment' , 
                                                       'score_pd'])
    print('Done calculating mean of columns ')
    
    print('Calculating sum of columns')
    df = calculate_sum_of_column(df, col_list=['qty_address_mutations_month',
                                                        'qty_started_names',
                                                         'qty_stopped_names',
                                                        'total_changeof_board_members_']) 
    print('Done calculating sum of columns')
    
    print('Calculating variance of columns')
    df = calculate_variance_of_column(df, col_list=['qty_employees',
                                                        'qty_issued_credit_reports',
                                                         'score_payment_assessment',
                                                        'score_pd']) 
    print('Done calculatinh variance of columns')


 
    print('Dropping old columns')
    df = drop_old_columns(df, col_list = ['date_established', 'year_consolidated_operating_result', 
                                          'year_consolidated_revenue', 
                                          'year_operating_result', 'year_qty_employees', 'year_revenue',
                                          'is_discontinued', 'code_financial_calamity', 
                                          'amt_consolidated_operating_result', 'amt_consolidated_revenue', 
                                          'amt_operating_result','amt_revenue','qty_employees', 
                                          'qty_issued_credit_reports', 'score_payment_assessment' ,
                                          'score_pd', 'color_credit_status','rat_pd',
                                          'qty_address_mutations_month','qty_started_names',
                                          'qty_stopped_names', 'total_changeof_board_members_', 'is_sole_proprietor',
                                          'code_discontinuation','date_financial_calamity_started', 
                                          'date_financial_calamity_stopped', 'id_company_creditproxy', 
                                          'financial_calamity_outcome', 'has_increased_risk' , 
                                          'perc_credit_limit_adjustment',
                                          'date_start', 'from_date_start', 'qty_address_mutations_total',
                                          'code_legal_form_group', 'code_SBI_2_group' ])
    print('Done dropping old columns')
	
	# Deduplicating rows of original dataframe
	df = deduplicate_rows(df)
    # Done deduplicating rows of original dataframe
	
	# Getting target of next year and adding it as a column
	df = replace_has_relocated_with_nextyear(df= one_year_df, next_year='2018',
                                   dir_prefix= 'including_scores/merged_per_year/merged_cleaned/relocation_dates')

	# Done getting target of next year and adding it as a column
	
    print('Done aggreating dataframe')
	
	if save_df_locally:
	    # Saving DF local to VM into files_to_bucket folder
	    save_df_locally(df= df, dir_prefix= 'files_to_bucket', year= '2017')
		
    return df

def save_df_locally(df, dir_prefix, year, as_json= False):
    """ Saves df as json or csv locally on server """
    if as_json:        
        file_path = dir_prefix + '/' + year + '_merged_cleaned.json'
        df.to_json(file_path)
    else:
        file_path =  dir_prefix + '/' + year + '_merged_cleaned.csv'
        df.to_csv(file_path)





