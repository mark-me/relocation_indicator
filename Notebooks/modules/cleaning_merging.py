import pandas as pd
import numpy as np
import gcsfs
from google.cloud import storage
from common import *

class Cleaner_Merger(GC_Data_Processing):

  def __init__(self, name_project, name_bucket, dir_input_data, dir_output_data):
      """Returns a new Cleaner Merger object."""
      GC_Data_Processing.__init__(self, name_project, name_bucket)
      self.dir_input_data = dir_input_data
      self.dir_output_data = dir_output_data
      self.columns_targets = ['date_month', 'id_company', 'id_branch', 'is_sole_proprietor', 'has_relocated']
      self.columns_features = ['date_month', 'id_company', 'id_branch',
                               'is_discontinued', 'financial_calamity_outcome', 'date_established', 'qty_employees', 
                               'year_qty_employees', 'id_company_creditproxy', 'score_payment_assessment', 
                               'amt_revenue', 'year_revenue', 'amt_consolidated_revenue', 'year_consolidated_revenue',
                               'amt_consolidated_operating_result', 'year_consolidated_operating_result', 
                               'perc_credit_limit_adjustment', 'color_credit_status', 'rat_pd', 'score_pd',
                               'has_increased_risk', 'is_sole_proprietor', 'code_SBI_2', 'code_SBI_1',
                               'qty_address_mutations_total', 'qty_address_mutations_month', 'has_name_change', 
                               'code_discontinuation', 'code_financial_calamity', 'qty_issued_credit_reports', 
                               'Associate', 'Authorized official', 'Board member', 'Chairman', 'Commissioner', 
                               'Director', 'Liquidator', 'Major', 'Managing clerk', 'Managing partner', 
                               'Member of the partnership', 'Miscellaneous', 'Owner', 'Secretary', 'Secretary/Treasurer', 
                               'Treasurer', 'Unknown', 'Vice President', 'amt_operating_result', 'code_legal_form', 
                                'date_financial_calamity_started', 'date_financial_calamity_stopped', 'date_start', 
                                'from_date_start', 'qty_stopped_names', 'qty_started_names', 'year_operating_result']   

  def month_delta(self, date, delta):
      """Adding/subtracting months to/from a date."""
      m, y = (date.month + delta) % 12, date.year + ((date.month) + delta - 1) // 12
      if not m: m = 12
      d = min(date.day, [31, 29 if y%4==0 and not y%400==0 else 28,31,30,31,30,31,31,30,31,30,31][m-1])
      return date.replace(day=d,month=m, year=y)

  def month_last_date(self, dtDateTime):
      dYear = dtDateTime.strftime("%Y")                       # Get the year
      dMonth = str(int(dtDateTime.strftime("%m"))%12+1)       # Get next month, watch rollover
      dDay = "1"                                              # First day of next month
      nextMonth = mkDateTime("%s-%s-%s"%(dYear,dMonth,dDay))  # Make a datetime obj for 1st of next month
      delta = datetime.timedelta(seconds=1)                   # Create a delta of 1 second
      return nextMonth - delta                                # Subtract from nextMonth and return

  def get_month_filenames(self, df_date_months):
      """ Get the file names of number of the months in the data frame """
      month_files = [] # List of month files

      df_date_months['year'] = df_date_months.date_month.dt.year
      list_years = df_date_months['year'].unique()
      # If there are multiple years, iterate through years
      for year in list_years:
          # Get the year's data file names
          dir_data_year = self.dir_input_data + '/' + str(year)
          list_blob = list(self.gc_bucket.list_blobs(prefix=dir_data_year))
          # finding out which month files should be processed by looking which contain the first month date (YYYY-mm-01)
          df_year_months = df_date_months[df_date_months['year'] == year]['date_month']
          for blob in list_blob:
              for month in df_year_months:
                  if (month.strftime("%Y-%m-%d") in blob.name) & ('CSV' in blob.name):
                      month_files.append(blob.name)
      return(month_files)

  def clean_data(self, df):
      """Cleans data and returns formatted df"""
      df['date_month'] = pd.to_datetime(df['date_month'])
      df['financial_calamity_outcome'] = df['financial_calamity_outcome'].fillna(-1)
      df['qty_employees'] = df['qty_employees'].str.strip()
      df.loc[df.qty_employees == 'NA', 'qty_employees'] = np.NaN
      df['year_qty_employees'] = df['year_qty_employees'].str.strip()
      df.loc[df.year_qty_employees == 'NA', 'year_qty_employees'] = np.NaN
      df['amt_revenue'] = df['amt_revenue'].str.strip()
      df.loc[df.amt_revenue == 'NA', 'amt_revenue'] = np.NaN
      df['year_revenue'] = df['year_revenue'].str.strip()
      df.loc[df.year_revenue == 'NA', 'year_revenue'] = 0
      df['amt_consolidated_revenue'] = df['amt_consolidated_revenue'].str.strip()
      df.loc[df.amt_consolidated_revenue == 'NA', 'amt_consolidated_revenue'] = np.NaN
      df['amt_consolidated_revenue'] = df['amt_consolidated_revenue'].astype(str).str.replace(',','.')
      df['year_consolidated_revenue'] = df['year_consolidated_revenue'].str.strip()
      df.loc[df.year_consolidated_revenue == 'NA', 'year_consolidated_revenue'] = np.NaN
      df['amt_consolidated_operating_result'] = df['amt_consolidated_operating_result'].str.strip()
      df.loc[df.amt_consolidated_operating_result == 'NA', 'amt_consolidated_operating_result'] = np.NaN
      df['amt_consolidated_operating_result'] = df['amt_consolidated_operating_result'].astype(str).str.replace(',','.')
      df['year_consolidated_operating_result'] = df['year_consolidated_operating_result'].str.strip()
      df.loc[df.year_consolidated_operating_result == 'NA', 'year_consolidated_operating_result'] = np.NaN
      df['score_pd'] = df['score_pd'].str.strip()
      df.loc[df.score_pd == 'NA', 'score_pd'] = np.NaN
      df['score_pd'] = df['score_pd'].astype(str).str.replace(',','.')
      df['has_increased_risk'] = df['has_increased_risk'].astype(bool)
      df.loc[df.date_established < '1700-12-31' , 'date_established'] = np.NaN
      df['date_established'] = pd.to_datetime(df['date_established'])
      df['amt_operating_result'] = df['amt_operating_result'].str.strip()
      df.loc[df.amt_operating_result == 'NA', 'amt_operating_result'] = np.NaN
      df['amt_operating_result'] = df['amt_operating_result'].astype(str).str.replace(',','.')
      df['year_operating_result'] = df['year_consolidated_operating_result'].str.strip()
      df.loc[df.year_operating_result == 'NA', 'year_operating_result'] = np.NaN
      return df

  def aggregate_board_members(self, df):
      """Agregates the number of board members into one feature """
      list_columns_sum = ['associate', 'authorized_official', 'board_member', 'chairman', 'commissioner',
                          'director', 'liquidator', 'major', 'managing_clerk', 'managing_partner',
                          'member_of_the_partnership', 'miscellaneous', 'owner', 'secretary',
                          'secretary/treasurer', 'treasurer', 'unknown', 'vice_president']
      df['total_changeof_board_members_'] = df[list_columns_sum].sum(axis=1)
      df = df.drop(columns=list_columns_sum)
      return df

  def get_relocation_dates(self):
      """ Reading relocation data of previous addresses."""
      df = self.get_df_from_bucket(self.dir_input_data + "/additional_data/location_start_date.CSV",
                                   sep=',', na_values=['', '1198-06-12', 'NA'],
                                   parse_dates=['date_relocation_last', 'date_relocation_penultimate'])
      print("Read relocation data with", df.shape[0], "rows and", df.shape[1], "columns.")
      return(df)

  def add_previous_relocation_dates(self, df_year_months):
      # Getting the relocation data
      df_relocation_dates = self.get_relocation_dates()
      # Getting all company, branch, month combinations
      df_branch_months = df_year_months[['id_company', 'id_branch', 'date_month']]
      df_branch_months = df_branch_months.merge(df_relocation_dates, 
                                                on=['id_company', 'id_branch'], 
                                                how='left')
      # Removing all relocation dates after the current month
      df_branch_months = df_branch_months[df_branch_months['date_month'] > df_branch_months['date_relocation_last']]
      # Getting the latest relocation dates
      df_max_dates = df_branch_months.groupby(['id_company', 'id_branch', 'date_month'])['date_relocation_last', 'date_relocation_penultimate'].max()
      df_max_dates = df_max_dates.reset_index()
      # Calculate years since relocations
      df_max_dates['years_current_location'] = (df_max_dates.date_month - df_max_dates.date_relocation_last)/np.timedelta64(1, 'Y')
      df_max_dates['years_previous_location'] = (df_max_dates.date_month - df_max_dates.date_relocation_penultimate)/np.timedelta64(1, 'Y')

      # Adding the new data to the original year data
      df_year_months = df_year_months.merge(df_max_dates,
                                            on=['id_company', 'id_branch', 'date_month'],
                                            how='left')
      return(df_year_months)

  def get_features(self, date_month, qty_months_horizon=12):
      """ Getting the features set """
      df_months_combined = pd.DataFrame()  # The data frame which will contain all independent variables
      month_files = []                     # List of month files in scope

      # Get all months
      date_start = self.month_delta(date_month, -qty_months_horizon)
      df_date_months = pd.DataFrame(pd.date_range(date_start, periods=qty_months_horizon, freq="M").tolist(),
                                    columns=['date_month'])
      df_date_months['date_month'] = df_date_months['date_month'].values.astype('datetime64[M]') # First day of month

      month_files = self.get_month_filenames(df_date_months) # Get the file names of all required month files

      # Cleaning, transforming and combining month files
      for month_file in month_files:
          with self.gc_fs.open('graydon-data/' + month_file) as f:
              df_month = pd.read_csv(f, sep=';', usecols= self.columns_features, index_col=False, nrows = 5000)
              print('Read', month_file, "with", df_month.shape[0], "rows and", df_month.shape[1], "columns")
              df_month = df_month[(df_month['is_sole_proprietor'] == 0)] 
              print('After removing sole proprietors there are', df_month.shape[0], "rows are left")
              df_month.columns = (df_month.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', ''))
              df_month = self.aggregate_board_members(df_month)
              df_month = self.clean_data(df_month)
              df_months_combined = df_months_combined.append(df_month)
              print('The number of rows so far by adding', month_file, ":", df_months_combined.shape[0])

      df_months_combined = self.add_previous_relocation_dates(df_months_combined)
      df_months_combined['date_dataset'] = date_month # Add the identifier for a data-set

      return(df_months_combined)

  def get_targets(self, date_month, qty_months_horizon=12):
      """ Getting the target (related) variable set """
      df_months_combined = pd.DataFrame()  # The data frame which will contain all independent variables

      # Get all months in range
      df_date_months = pd.DataFrame(pd.date_range(date_month, periods=qty_months_horizon, freq="M").tolist(),
                                    columns=['date_month'])
      df_date_months['date_month'] = df_date_months['date_month'].values.astype('datetime64[M]') # First day of month

      # Get the file names of all required month files
      month_files = self.get_month_filenames(df_date_months)

      # Cleaning, transforming and combining month files
      for month_file in month_files:
          with self.gc_fs.open('graydon-data/' + month_file) as f:
              df_month = pd.read_csv(f, sep=';', usecols= self.columns_targets, index_col=False, nrows = 5000)  
              print('Read', month_file, "with", df_month.shape[0], "rows")
              df_month = df_month[(df_month['is_sole_proprietor'] == 0)] # & (one_month_df['is_discontinued'] == 0)
              print('After removing sole proprietors there are', df_month.shape[0], "rows are left")
              df_month.columns = (df_month.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', ''))
              df_months_combined = df_months_combined.append(df_month)
              print('The number of rows so far by adding ', month_file, ":", df_months_combined.shape[0])

      df_months_combined['date_dataset'] = date_month # Add the identifier for a data-set

      # Aggregating data to year
      df_months_combined = df_months_combined.groupby(['date_dataset', 
                                                       'id_company', 
                                                       'id_branch']).agg({'has_relocated': 'max', 
                                                                          'date_month': 'max'})
      df_months_combined = df_months_combined.rename(index=str, columns={"date_month": "date_month_last"})
      df_months_combined = df_months_combined.reset_index()
      df_months_combined['date_dataset'] = pd.to_datetime(df_months_combined['date_dataset'])
      df_months_combined['id_company'] = df_months_combined['id_company'].astype(int)
      df_months_combined['id_branch'] = df_months_combined['id_branch'].astype(int)

      return(df_months_combined)

  def combine_features_target(df_features, df_target):
      """Combining features and target."""
      df_monthly = df_features.merge(df_target,
                                     on=['id_company', 'id_branch'],
                                     how='left')
      bool_na_relocation = df_monthly['has_relocated']

  def clean_merge_data(self, date_month):
      """Collecting all input data to form a montly data file."""
      print("Reading and cleaning features")
      df_features = self.get_features(date_month, self.columns_features)
      print("Reading and cleaning target")
      df_target = self.get_targets(date_month, self.columns_targets)
      print("Combining feature and target data")
      df_dataset = self.combine_features_target(df_features, df_target)

      if not os.path.exists("data_out"):
          os.mkdir("data_out")
      file_output = "cleaned_merged_" + date_dataset.strftime('%Y-%m-%d') + ".csv"
      print("Saving merged and cleaned data to", file_output)
      self.save_df_locally(df_dataset, self.dir_output_data, file_output)
      print("Copying file to bucket to", self.dir_output_data)
      self.local_file_to_bucket(file_source = data_out + "/" + file_output,
                                dir_bucket = self.dir_output_data)