import os
import yaml
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats.mstats import winsorize 



FILENAME = "loan_payments.csv"

def load_database_credentials(filename):
    with open(filename,'r') as file:
        return yaml.safe_load(file)
    
def load_local_df(filename):
    return pd.read_csv(filename)

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


class DataTransform:
    def __init__(self, df):
        self.df = df
        
    def convert_to_float(self, col_name):
        """
        Convert the specified column to float type by extracting numerical values from strings.
        """
        self.df[col_name] = self.df[col_name].str.extract('(\d+)').astype(float)
        
    def convert_to_datetime(self, col_name):
        """
        Convert the specified column to a datetime type.
        """
        self.df[col_name] = pd.to_datetime(self.df[col_name])   # convert to datetime
        
    def set_datetime_display_format(self, format_str='%b-%Y'):
        """
        Set the display format for datetime columns in the DataFrame.

        Parameters:
        - format_str (str): The format string for datetime display, e.g., '%b-%Y' for 'Dec-2021'.
        """
        pd.set_option('display.date_yearfirst', False)
        pd.set_option('display.date_dayfirst', False)
        pd.set_option('display.datetime_format', format_str)

    def convert_to_categorical(self, col_name):
        """
        Convert the specified column to a categorical type.
        """
        self.df[col_name] = self.df[col_name].astype('category')



class DataFrameInfo:
    def __init__(self, df):
        self.df = df

    def info(self):
        """
        Print information about the DataFrame, including the number of non-null entries and the data type of each column.
        """
        return self.df.info()

    
    def describe(self):
        """
        Return descriptive statistics summarizing the central tendency, dispersion, and shape of the dataset's distribution, excluding NaN values.
        """
        return self.df.describe()

    # Plots correlation matrix for all numerical values in the DataFrame 
    def plot_correlation_matrix(self, threshold=0.5):
        """
        Generate a heatmap representing the correlation matrix of the DataFrame.
        A column is displayed in the matrix if it has a correlation value higher than the threshold 
        with one or more other columns.

        Parameters:
        - threshold (float): Threshold for displaying columns based on their correlation values. Only columns with 
        at least one correlation above this absolute value are displayed.
        """

        # Styling
        plt.rc("axes.spines", top=False, right=False)
        sns.set_style('darkgrid')
        five_thirty_eight = ["#30a2da", "#fc4f30", "#e5ae38", "#6d904f", "#8b8b8b"]
        sns.set_palette(five_thirty_eight)

        # Compute the correlation matrix and round to two decimal places
        corr = self.df.corr().round(2)

        # Filter out correlations below a certain threshold (absolute value)
        filtered_corr = corr[(corr.abs() >= threshold).any(axis=1)].dropna(axis=1, how='all')

        # Generate a mask for the upper triangle
        mask = np.zeros_like(filtered_corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        # Plotting
        plt.figure(figsize=(15, 15))
        sns.heatmap(filtered_corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True), 
                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75}, vmin=-1, vmax=1)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45)
        plt.title(f'Correlation Matrix of Numerical Variables with |Correlation| ≥ {threshold}')
        plt.show()

    def column_correlation(self, column):
        """
        Print the 10 highest correlations for a specified column.

        Parameters:
        - column (str): The column name for which correlations are to be printed.
        """
        correlation = self.df.corr()[column].sort_values(ascending=False).head(10)
        print(correlation)

    def count_column_distinct_values(self, col_name):
        """
        Count and return distinct values in a specified categorical column.

        Parameters:
        - col_name (str): Name of the column to count distinct values.
        """
        return self.df[col_name].value_counts(dropna=False)

    def shape(self):
        """
        Return the shape (dimensions) of the DataFrame.
        """
        return self.df.shape

    def count_nan(self):
        """
        Generate and return a DataFrame containing the count and percentage of NULL values in each column.
        """
        count_nan = self.df.isna().sum()
        percent_nan = (self.df.isna().sum() / len(self.df)) * 100
        return pd.DataFrame({
            'count_nan': count_nan,
            'percent_nan': percent_nan
        })

    
    def column_info(self, column_names):
        """
        Calculate and print the Interquartile Range (IQR), bounds for outliers, and other statistics for specified columns.

        Parameters:
        - column_names (list): List of column names for which statistics are to be printed.
        """
        for column in column_names:
            # Calculating the Q1, Q3, and IQR
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1

            # Define the bounds for outliers
            lower_bound = max(Q1 - 1.5 * IQR, 0)  # Adjusted lower bound to be non-negative
            upper_bound = Q3 + 1.5 * IQR
            
            # Calculate IQR for non-zero data
            non_zero_data = self.df[self.df[column] > 0]
            Q1_non_zero = non_zero_data[column].quantile(0.25)
            Q3_non_zero = non_zero_data[column].quantile(0.75)
            IQR_non_zero = Q3_non_zero - Q1_non_zero
            lower_bound_non_zero = max(Q1_non_zero - 1.5 * IQR_non_zero, 0)
            upper_bound_non_zero = Q3_non_zero + 1.5 * IQR_non_zero
            
            # Find outliers
            outliers_condition = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound))
            outliers_data = self.df[column][outliers_condition].sort_values()
            
            print(f"Column: {column}\n"
                    f"The IQR range for {column} is: {lower_bound:.2f} - {upper_bound:.2f}\n"
                    f"Median: {self.df[column].median()}\n"
                    f"The IQR range for non-zero data in {column} is: {lower_bound_non_zero:.2f} - {upper_bound_non_zero:.2f}\n"
                    f"Median for non-zero data: {non_zero_data[column].median()}\n"
                    f"Outliers for {column}: {outliers_data}\n"
                    f"Number of outliers: {outliers_data.shape[0]}\n"
                    f"Percentage of outliers: {outliers_data.shape[0] * 100 / self.df.shape[0]:.2f}%\n")

class Plotter:
    def __init__(self):
        pass


    def plot_missing_values(self, df, color: str = "#fc4f30", rotation: int = 90):
        """
        Generate a bar plot showing count of missing values for each column in df.

        Parameters:
        - df (pd.DataFrame): DataFrame to analyze.
        - color (str): Color of the bars in the plot.
        - rotation (int): Rotation angle for x-axis labels.
        """
        missing_values = df.isnull().sum()
        plt.figure(figsize=(15, 8))
        bars = plt.bar(missing_values.index, height=missing_values.values, color=color)
        plt.title('Count of NULL values per column')
        plt.ylabel('Number of NULL values')
        plt.xlabel('Columns')
        plt.xticks(rotation=rotation)

        # Add grid lines
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

        max_height = len(df)  # Set max_height to total number of rows
        ylim_top = 1.1 * max_height  # Set y-axis upper limit to be 110% of the maximum possible missing values

        # Add the actual number of missing values on top of each bar
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Exclude columns with no missing values
                plt.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation='vertical')
        
        plt.ylim(0, ylim_top)  # Adjust the y-axis upper limit
        plt.tight_layout()  # Adjust layout for better appearance
        plt.show()
                
    def plot_skewness(self, df, columns):
        """
        Generate plots to show the skewness of the specified columns in the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data.
        - columns (list): The list of column names to analyze for skewness.
        """

        sns.set(font_scale=0.7)
        f = pd.melt(df, value_vars=columns)
        g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)

        # Add annotations with skewness values
        for ax, title in zip(g.axes.flat, columns):
            skew_val = df[title].skew()
            ax.text(0.95, 0.95, f"Skew: {skew_val:.2f}",
                    verticalalignment='top', horizontalalignment='right',
                    transform=ax.transAxes, color='red', fontsize=10)

        plt.show()

    def plot_skewedness_with_log(self, df, columns):
        """
        Generate plots to show skewness and log-transformed skewness in the specified columns of the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data.
        - columns (list): The list of column names to analyze for skewedness.
        """
        # Dataframe for log-transformed values
        log_df = df[columns].applymap(lambda x: np.log1p(x))

        sns.set(font_scale=0.7)
        melted_df = pd.melt(df, value_vars=columns)

        # Set up the FacetGrid for plotting
        g = sns.FacetGrid(melted_df, col="variable", col_wrap=3, sharex=False, sharey=False)
        
        # Plotting function
        def dual_hist(x, **kwargs):
            ax = plt.gca()
            sns.histplot(x, ax=ax, kde=True, label="Original", color="blue")
            sns.histplot(np.log1p(x), ax=ax, kde=True, label="Log Transformed", color="orange")
            ax.legend()

        # Map the plotting function
        g.map(dual_hist, "value")

        # Add annotations with skewness values for original and log-transformed data
        for ax, title in zip(g.axes.flat, columns):
            skew_val = df[title].skew()
            log_skew_val = log_df[title].skew()
            ax.text(0.95, 0.95, f"Skew: {skew_val:.2f}\nLog Skew: {log_skew_val:.2f}",
                    verticalalignment='top', horizontalalignment='right',
                    transform=ax.transAxes, color='red', fontsize=10)

        plt.show()


    def plot_outliers(self, df, columns):
        """
        Generate boxplots to show outliers in the specified columns of the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data.
        - columns (list): The list of column names to analyze for outliers.
        """
        sns.set(font_scale=0.7)
        f = pd.melt(df, value_vars=columns)
        g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False)
        g = g.map(sns.boxplot, "value", orient="v", boxprops=dict(alpha=.3))
        g.set_titles(col_template="{col_name}")
        plt.show()


    def plot_std_dev_outliers(self, df, columns):
        """
        Generate scatter plots to identify potential outliers in the specified columns of a DataFrame. 
        Outliers are identified based on the standard deviation from the mean.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data.
        - columns (list): List of column names to be analyzed for outliers.
        """
        num_columns = len(columns)
        
        # Calculate the number of rows needed (every row can have up to 3 columns)
        num_rows = num_columns // 3 + (num_columns % 3 > 0)
        
        # Create a figure with a subplot for each variable
        fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows), squeeze=False)  # `squeeze=False` makes sure axs is always a 2D array
        
        # Flatten the array of axes for easy iteration
        axs = axs.flatten()

        # Adjust font sizes if necessary
        plt.rc('font', size=11)  # Adjust font size here

        if num_columns == 1:
            axs = [axs]  # If only one column, wrap it in a list so iteration works

        for idx, column in enumerate(columns):
            ax = axs[idx]  # Get the corresponding axis object
            mean = df[column].mean()
            std_dev = df[column].std()

            # Define the bounds for outliers
            bounds = [mean - 3 * std_dev, mean + 3 * std_dev]

            # Create a range of x-values to spread data points evenly
            x_values = np.arange(len(df))

            # Plot the points and mark the mean and bounds
            ax.scatter(x_values, df[column], alpha=0.6)  # plot the data points with equal horizontal spacing
            ax.axhline(mean, color='green', linestyle='--', label='Mean')
            ax.axhline(bounds[0], color='red', linestyle='--', label='-3 STD')
            ax.axhline(bounds[1], color='red', linestyle='--', label='+3 STD')

            # Highlight potential outliers outside of 3 STD
            outliers_condition = (df[column] < bounds[0]) | (df[column] > bounds[1])
            ax.scatter(x_values[outliers_condition], df[column][outliers_condition], color='red', label='Potential Outlier')

            ax.set_title(f'SD Outliers: {column}')
            ax.set_xlabel('Data Point Index')
            ax.set_ylabel(column)
            ax.legend()

        plt.tight_layout(pad=2.0)
        plt.show()

    def plot_kde(self, df, column, non_zero=False):
        """
        Generate a Kernel Density Estimate (KDE) plot for a specified column in a DataFrame. 
        Optionally, the plot can be generated only for non-zero values.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data.
        - column (str): The name of the column to be plotted.
        - non_zero (bool): If True, the KDE plot will be generated only for non-zero values.
        """
        if non_zero:
            sns.kdeplot(df[df[column] > 0][column], cut=0, shade=True)
            plt.title(f'KDE Plot of Non-Zero {column}')
            plt.xlabel(column)
        else:
            sns.kdeplot(df[column], cut=0, shade=True)
            plt.title(f'KDE Plot of {column}')
            plt.xlabel(column)
        plt.show()

    def plot_categorical_distribution(self, df, column_name, color='skyblue', rotation=45):
        """
        Generate a bar plot to show the distribution of a categorical column in the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data.
        - column_name (str): The name of the categorical column to be plotted.
        - color (str): Color of the bars in the plot. Default is 'skyblue'.
        - rotation (int): Rotation angle for x-axis labels. Default is 45 degrees.
        """
        # Count the occurrences of each unique category
        value_counts = df[column_name].value_counts()

        # Create the bar plot
        plt.bar(value_counts.index, value_counts.values, color=color)

        # Add labels and title
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.title('Bar Plot of ' + column_name)

        # Rotate x-axis labels for better readability (optional)
        plt.xticks(rotation=rotation)

        # Display the bar plot
        plt.show()

   
class DataFrameTransform:   # transforming dataframe
    def __init__(self, df):
        self.df = df

    def impute_missing_values(self, column_name, method='median'):  
        """
        Impute missing values in a specified column using the specified method.

        Parameters:
        - column_name (str): Name of the column in which missing values are to be imputed.
        - method (str): Method to use for imputation ('median' or 'mean'). Default method is 'median'
        """
        if method == 'median':
            impute_value = self.df[column_name].median()
        elif method == 'mean':
            impute_value = self.df[column_name].mean()
        else:
            raise ValueError("Invalid imputation method. Use 'median' or 'mean'.")

        self.df[column_name].fillna(impute_value, inplace=True)

    
    def impute_categorical_proportionally(self, column_name):
        """
        Impute missing values in a specified categorical column proportionally 
        based on the distribution of the existing values.

        Parameters:
        - column_name (str): The name of the categorical column where missing values will be imputed.
        """
        # 1. Compute Probabilities
        value_counts = self.df[column_name].value_counts(normalize=True)

        # 2. Determine Missing Rows
        missing_indices = self.df[self.df[column_name].isna()].index

        # 3. Assign Values Based on Probabilities
        imputed_values = np.random.choice(value_counts.index, p=value_counts.values, size=len(missing_indices))
        self.df.loc[missing_indices, column_name] = imputed_values

    
    def log_transform(self, column_names):
        """
        Apply a logarithmic transformation to specified columns of the DataFrame.

        Parameters:
        - column_names (list): List of column names to apply the logarithmic transformation to.
        
        Returns:
        - pd.DataFrame: The transformed DataFrame.
        """

        log_df = self.df[column_names].applymap(lambda x: np.log1p(x) if x > 0 else 0)
        self.df[column_names] = log_df
        return self.df

    
    def sqrt_transform(self, column_names):
        """
        Apply a square root transformation to specified columns of the DataFrame.

        Parameters:
        - column_names (list): List of column names to apply the square root transformation to.
        
        Returns:
        - pd.DataFrame: The transformed DataFrame.
        """
        sqrt_df = self.df[column_names].applymap(lambda x: np.sqrt(x) if x > 0 else 0)
        self.df[column_names] = sqrt_df
        return self.df
    
    
    def remove_IQR_outliers(self, column_names):
        """
        Removes outliers from one or several columns based on IQR method.

        Parameters:
        - column_names (list): List of column names to remove outliers from.
        """

        # Calculate the Q1, Q3, and IQR
        Q1 = self.df[column_names].quantile(0.25)
        Q3 = self.df[column_names].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Adjust lower bounds to be non-negative for count data
        lower_bound[lower_bound < 0] = 0
        
        # Remove outliers
        for column in column_names:
            self.df = self.df[(self.df[column] >= lower_bound[column]) & (self.df[column] <= upper_bound[column])]

        
    def remove_custom_outliers(self, column, lower_bound=None, upper_bound=None):
        """
        Removes outliers from a specified column using custom bounds or the Interquartile Range (IQR) method.

        Parameters:
        - column (str): Name of the column to remove outliers from.
        - lower_bound (float, optional): Custom lower bound for defining outliers. If None, calculated using IQR.
        - upper_bound (float, optional): Custom upper bound for defining outliers. If None, calculated using IQR.
        """
        if lower_bound is None or upper_bound is None:
            # Calculate the Q1, Q3, and IQR
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Calculating lower and upper bounds or using the values provided
            lower_bound = Q1 - 1.5 * IQR if lower_bound is None else lower_bound
            upper_bound = Q3 + 1.5 * IQR if upper_bound is None else upper_bound
        
        # Remove outliers
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]    
        
    
    def winsorize_outliers(self, column_names, limits=(0.05, 0.05)):
        """
        Apply winsorization to limit extreme values in a DataFrame column.

        Parameters:
        - column_name (str): Name of the column to be winsorized.
        - limits (tuple): The proportion of data to be replaced on both tails. For example, (0.05, 0.05) for both ends.
        """
        for column in column_names:
            self.df[column] = winsorize(self.df[column], limits=limits)


if __name__ == "__main__":
    credentials = load_database_credentials('credentials.yaml')
    db_connector = RDSDatabaseConnector(credentials)

    # Extract data from the loan_payments table
    df_loan_payments = db_connector.extract_data()

    # Save df to a csv
    db_connector.save_df(df_loan_payments, FILENAME)


    df = load_local_df(FILENAME)


        