# importing relevant Libraries
import pandas as pd
import numpy as np
import copy


# Cleaning the data
def CleanData(path:str):
    # reading the data
    df = pd.read_csv(path)
    
    # dropping rows that doesn't contain product description
    df = drop_nulls(df,'Description')
    
    # With some exploration of the stockcode data, I found out that the code that 
    # doesn't contain a number in it, does not have an actual product description.
    # Example 'M', 'S', 'AMAZONFEE' etc. We can get rid of that data as well
    df['StockCode'] = df['StockCode'].astype('str').str.extractall('(\d+)')\
                    .unstack().fillna('').sum(axis=1).astype(int)
    
    #dropping the stockcodes that became null after extracting numerical codes, to remove wrong data
    df = drop_nulls(df,'StockCode')
    
    # Taking only United Kingdom data for simplicity as it contains 91% of the data
    df = df.loc[df.Country=='United Kingdom'].reset_index(drop=True)
    # Dropping unneccessary columns
    df = df.drop(['Country','InvoiceNo'],axis=1)
    # Changing type of date to datetime
    df.InvoiceDate = pd.to_datetime(df.InvoiceDate,format = '%m/%d/%Y %H:%M')
    
    return df

# Dropping Null values
def drop_nulls(df, col:str):
    df = copy.deepcopy(df)
    index_to_drop = np.where(df[col].isna())[0]
    df = df.drop(index_to_drop,axis=0).reset_index(drop=True)
    return df


if __name__ == '__main__':
    df = CleanData('Online_Retail.csv')
    df.to_csv("cleaned_online_retail.csv")