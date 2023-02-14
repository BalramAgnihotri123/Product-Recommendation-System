import pandas as pd
import numpy as np

import copy
import datetime
import warnings
warnings.filterwarnings("ignore")

import dask.dataframe as dd
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

class RecommendationEngine:       
    def __init__(self, root: str, lower_bound=0.7, n_recommendations=20, date_format='%d/%m/%Y %H:%M'):
        self.stop_words = set(stopwords.words('english'))
        self.lower_bound = lower_bound
        self.n_recommendations = n_recommendations
        self.date_format = date_format
        self.root = root

    def remove_stop_words(self, text):
        words = text.split()
        words = [word for word in words if word.lower() not in self.stop_words]
        return " ".join(words)

    def drop_nulls(self, df, col: str):
        df = copy.deepcopy(df)
        index_to_drop = np.where(df[col].isna())[0]
        df = df.drop(index_to_drop,axis=0).reset_index(drop=True)
        return df

    
    def divide_data(self, df, cust_id: float, date_str):
        '''
        Divides data on the basis of the customer id and all the products that the customer purchased 
        before a particular date...
        returns two dataframes one of which is just the data mentioned above and other is the full dataframe 
        '''

        df_total = copy.deepcopy(df)

        df = copy.deepcopy(df)
        df = self.drop_nulls(df, 'CustomerID')
        df = df.sort_values('CustomerID' , ascending=True)
        # use the rank() method to convert the values to a 1-based ranking
        df['CustomerID'] = df['CustomerID'].rank(method='dense').astype(int)


        if len(df.loc[(df.CustomerID == cust_id)]) == 0:
            raise ValueError("Invalid Customer ID")

        try:
            date = datetime.datetime.strptime(date_str, self.date_format)
        except ValueError as e:
            try:
                # Try to parse just the date portion
                date = datetime.datetime.strptime(date_str, '%d/%m/%Y')
                # Assume a default time of 00:00 if the time is not specified
                date = date.replace(hour=0, minute=0)
            except ValueError as e:
                raise ValueError(f"Incorrect date format. Expected format: {self.date_format}") from e

        df_cust = df.loc[(df.CustomerID == cust_id) & (df.InvoiceDate < date)].reset_index(drop=True)
        return df_cust, df_total 
    
    # To part a string into combinations of n
    def ngrams(self,
               string,
               n=3):
        '''
        Returns a list of all combinations of n consecutive letters
        for a given string.
        Args:
            string (str): A given string
            n      (int): The number of letters to use
        Returns:
            list: all n letter combinations of the string
        '''
        string = string.upper()
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]


    def awesome_cossim_top(self, A, B, ntop, lower_bound=0):
        # To calculate Cosine similarity in two vectorized string 
        # force A and B as a CSR matrix.
        # If they have already been CSR, there is no overhead
        A = A.tocsr()
        B = B.tocsr()
        M, _ = A.shape
        _, N = B.shape

        idx_dtype = np.int32

        nnz_max = M*ntop

        indptr = np.zeros(M+1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)

        return csr_matrix((data,indices,indptr),shape=(M,N))
    
    
    def get_matches_df(self, similarity_matrix, A, B):
        # getting best matches with given similarity
        '''
        Takes a matrix with similarity scores and two arrays, A and B,
        as an input and returns the matches with the score as a dataframe.
        Args:
            similarity_matrix (csr_matrix)  : The matrix (dimensions: len(A)*len(B)) with the similarity scores
            A              (pandas.Series)  : The array to be matched (dirty)
            B              (pandas.Series)  : The baseline array (clean)
        Returns:
            pandas.Dataframe : Array with matches between A and B plus scores
        '''
        non_zeros = similarity_matrix.nonzero()

        sparserows = non_zeros[0]
        sparsecols = non_zeros[1]

        nr_matches = sparsecols.size

        dirty = np.empty([nr_matches], dtype=object)
        clean = np.empty([nr_matches], dtype=object)
        similarity = np.zeros(nr_matches)

        dirty = np.array(A)[sparserows]
        clean = np.array(B)[sparsecols]
        similarity = np.array(similarity_matrix.data)

        df_tuples = list(zip(clean, dirty, similarity))

        return pd.DataFrame(df_tuples, columns=['bought_by_customer', 'recommendation', 'similarity'])
    
    def recommend_products(self, cust_id, date_str):
        df = pd.read_csv(self.root)
        df.InvoiceDate = pd.to_datetime(df.InvoiceDate,format = '%Y/%m/%d %H:%M:%S')
        
        df_cust, df_total =  self.divide_data(df, 
                                              cust_id,
                                              date_str)

        df_cust['Description'] = df_cust['Description'].map(self.remove_stop_words)
        df_total['Description'] = df_total['Description'].map(self.remove_stop_words)
        
        dfcust_des = dd.from_pandas(df_cust.Description, npartitions=1)
        dftotal_des = dd.from_pandas(df_total.Description,npartitions=6)
        
        ## Vectorizing the strings and to create n buckets (ngrams)
        try:
            vectorizer = TfidfVectorizer(min_df=1, analyzer=self.ngrams)
            tf_idf_matrix_dfcust_des_clean = vectorizer.fit_transform(dfcust_des)
            tf_idf_matrix_dftotal_des_dirty = vectorizer.transform(dftotal_des)
        except:
            raise ValueError("No Data of the customer before this Date.")
        
        csr_matrix = self.awesome_cossim_top(tf_idf_matrix_dftotal_des_dirty,
                                             tf_idf_matrix_dfcust_des_clean.transpose(),
                                             6,
                                             self.lower_bound)
        
        df_matched = self.get_matches_df(csr_matrix, dftotal_des, dfcust_des)
        df_matched = df_matched[~df_matched["recommendation"].isin(df_matched["bought_by_customer"])].sort_values('similarity',ascending=False)
        
        recommendations = df_matched.recommendation.unique()[:self.n_recommendations]
        
        return recommendations

# if __name__ == '__main__':
    # recom_engine = RecommendationEngine('Online_Retail.csv')
    # recommendations = recom_engine.recommend_products(13113,date_str='10/02/2023')
    # print(recommendations)