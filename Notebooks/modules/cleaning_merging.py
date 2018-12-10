def month_delta(date, delta):
    """Adding/subtracting months to/from a date."""
    m, y = (date.month + delta) % 12, date.year + ((date.month) + delta - 1) // 12
    if not m: m = 12
    d = min(date.day, [31, 29 if y%4==0 and not y%400==0 else 28,31,30,31,30,31,31,30,31,30,31][m-1])
    return date.replace(day=d,month=m, year=y)

def aggregate_board_members(df):
    """Agregates the number of board members into one feature """    
    col_list_to_sum = ['associate', 'authorized_official', 'board_member', 'chairman', 'commissioner',
                       'director', 'liquidator', 'major', 'managing_clerk', 'managing_partner',
                       'member_of_the_partnership', 'miscellaneous', 'owner', 'secretary',
                       'secretary/treasurer', 'treasurer', 'unknown', 'vice_president']  
    df['total_changeof_board_members_'] = df[col_list_to_sum].sum(axis=1)
    df = df.drop(columns=col_list_to_sum)
    return df
    
def get_relocation_dates(date_dataset):
    """ Reading relocation data of previous addresses."""
    blob_list = list(bucket.list_blobs(prefix='location_start_date.CSV'))

    for blob in blob_list: 
        with fs.open('graydon-data/' + blob.name) as f:
            df_relocation_dates = pd.read_csv(f, sep=',', 
                                              na_values=['', '1198-06-12', 'NA']) 
            df_relocation_dates['date_relocation_last'] = pd.to_datetime(df_relocation_dates['date_relocation_last'])
            df_relocation_dates['date_relocation_penultimate'] = pd.to_datetime(df_relocation_dates['date_relocation_penultimate'])
            
    return(df_relocation_dates)

def get_month_filenames(df_date_months, dir_data):
    """ Get the file names of number of the months in the data frame """
    month_files = [] # List of month files
    
    df_date_months['year'] = df_date_months.date_month.dt.year
    list_years = df_date_months['year'].unique()

    # If there are multiple years, iterate through years  
    for year in list_years:
        # Get the year's data file names
        dir_data_year = dir_data + '/' + str(year)
        list_blob = list(bucket.list_blobs(prefix=dir_data_year))

        # finding out which month files should be processed by looking which contain the first month date (YYYY-mm-01)
        df_year_months = df_date_months[df_date_months['year'] == year]['date_month']
        for blob in list_blob:
            for month in df_year_months:
                if (month.strftime("%Y-%m-%d") in blob.name) & ('CSV' in blob.name):
                    month_files.append(blob.name)
                    
    return(month_files)

def get_targets(date_month, columns_targets, dir_data):
    """ Getting the dependent variable set """
    df_months_combined = pd.DataFrame()  # The data frame which will contain all independent variables
    month_files = []                     # List of month files in scope
    
    # Get all months
    date_start = month_delta(date_month, -12)
    df_date_months = pd.DataFrame(pd.date_range(date_start, periods=12, freq="M").tolist(),
                                  columns=['date_month'])
    df_date_months['date_month'] = df_date_months['date_month'].values.astype('datetime64[M]') # First day of month
    
    # Get the file names of all required month files
    month_files = get_month_filenames(df_date_months, dir_data)
    
    # Cleaning, transforming and combining month files    
    for month_file in month_files:
        with fs.open('graydon-data/' + month_file) as f:
            df_month = pd.read_csv(f, sep=';', usecols= columns_targets, index_col=False)   
            df_month = df_month[(df_month['is_sole_proprietor'] == 0)] 
            df_month.columns = (df_month.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', ''))
            df_month = aggregate_board_members(df_month)
            df_month = clean_month_data(df_month)
            df_months_combined = df_months_combined.append(df_month)
            print('The number of rows so far by adding', month_file, ":", df_months_combined.shape[0])
            
    df_months_combined['date_dataset'] = date_month        
    
    return(df_months_combined)

def get_features(date_month, columns_features, dir_data):
    """ Getting the independent variable set """
    df_months_combined = pd.DataFrame()  # The data frame which will contain all independent variables
    
    # Get all months in range
    df_date_months = pd.DataFrame(pd.date_range(date_month, periods=12, freq="M").tolist(),
                                  columns=['date_month'])
    df_date_months['date_month'] = df_date_months['date_month'].values.astype('datetime64[M]') # First day of month

    # Get the file names of all required month files
    month_files = get_month_filenames(df_date_months, dir_data)
                    
    # Cleaning, transforming and combining month files                
    for month_file in month_files:
        with fs.open('graydon-data/' + month_file) as f:
            df_month = pd.read_csv(f, sep=';', usecols= columns_features, index_col=False)   
            df_month = df_month[(df_month['is_sole_proprietor'] == 0)] # & (one_month_df['is_discontinued'] == 0) 
            df_month.columns = (df_month.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', ''))
            df_months_combined = df_months_combined.append(df_month)
            print('The number of rows so far by adding ', month_file, ":", df_months_combined.shape[0])
     
    df_months_combined['date_dataset'] = date_month
    
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
