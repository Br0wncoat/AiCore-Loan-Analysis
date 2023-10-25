import os
import yaml
import pandas as pd
from sqlalchemy import create_engine

def load_database_credentials(filename):
    with open(filename,'r') as file:
        return yaml.safe_load(file)
    
def load_local_df(filename):
    with open (filename,'r') as file:
        df = pd.read_csv(filename)
    return df

class RDSDatabaseConnector:
    def __init__(self, credentials):
        self.credentials = credentials
        self.engine = None

    def init_engine(self):
        if self.engine is None:
            conn_str = f"postgresql+psycopg2://{self.credentials['RDS_USER']}:{self.credentials['RDS_PASSWORD']}@{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/{self.credentials['RDS_DATABASE']}"
            self.engine = create_engine(conn_str)
        return self.engine
    
    def extract_data(self, table_name="loan_payments"):
        """
        Extracts data from the specified table in the RDS database 
        and returns it as a Pandas DataFrame.
        """
        if self.engine is None:
            self.init_engine()

        query = f"SELECT * FROM {table_name};"
        return pd.read_sql_query(query, self.engine)

    def save_df(self, df, filename):
        """
        Saves the dataframe to the local machine
        """
        return df.to_csv(filename)



os.chdir('C:/Users/dmitr/PycharmProjects/AiCore-Loan-Analysis')

credentials = load_database_credentials('credentials.yaml')
db_connector = RDSDatabaseConnector(credentials)

# Extract data from the loan_payments table
df_loan_payments = db_connector.extract_data()

# Save df to a csv
db_connector.save_df(df_loan_payments, "loan_payments.csv")


df = load_local_df("loan_payments.csv")
print(df.head(5))
print(df.info)

        