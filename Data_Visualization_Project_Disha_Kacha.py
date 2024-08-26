#%%Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import shapiro, boxcox
import seaborn as sns
from prettytable import PrettyTable

#%%Read file
fashion_data = pd.read_csv('/Users/dishakacha/Downloads/Data_Visualization/Data Visualization/Project/mock_fashion_data_uk_us.csv')
print(fashion_data.head())

pd.set_option('display.precision', 2)
print(fashion_data.dtypes)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

#%%
table = PrettyTable()

# Add column names
table.field_names = fashion_data.columns

# Add rows for the head of the DataFrame
for row in fashion_data.head().itertuples(index=False):
    formatted_row = [f'{val:.2f}' if isinstance(val, (int, float)) else val for val in row]
    table.add_row(formatted_row)

# Print the table
print(table)
#%%

# Get summary statistics from describe
summary_stats = fashion_data.describe()

# Round the summary statistics to 2 decimal places
summary_stats = summary_stats.round(2)

# Create a PrettyTable object
table = PrettyTable()

# Add columns to the table
table.field_names = [''] + summary_stats.columns.tolist()  # Include an empty string for the index column

# Add rows to the table
for idx, row in summary_stats.iterrows():
    table.add_row([idx] + [f"{val:.2f}" if isinstance(val, float) else val for val in row])

# Print the table
print(table)
#%%Preprocessing checking null values
missing_values = fashion_data.isnull().sum()
print("Missing values in each column:\n", missing_values)

#%%Outlier Detection 1
columns_to_check = ['Price', 'Rating', 'Review Count', 'Age']

# Initialize a dictionary to hold outlier data
outliers_data = {}

# Compute IQR and identify outliers for each column
for column in columns_to_check:
    Q1 = fashion_data[column].quantile(0.25)
    Q3 = fashion_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtering outliers
    outliers = fashion_data[(fashion_data[column] < lower_bound) | (fashion_data[column] > upper_bound)]
    outliers_data[column] = outliers[[column]]

    # Print the outliers and check if there are any
    if not outliers.empty:
        print(f"Outliers in the '{column}' column:")
        print(outliers[column])
    else:
        print(f"No outliers detected in the '{column}' column.")

#%%Outlier Detection 2
fig, axes = plt.subplots(nrows=1, ncols=len(columns_to_check), figsize=(18, 6))

# Plot each column
for i, column in enumerate(columns_to_check):
    axes[i].boxplot(fashion_data[column].dropna(), patch_artist=True)  # dropna() to avoid errors with NaN
    axes[i].set_title(f'Box Plot of {column}')
    axes[i].set_ylabel(column)
    axes[i].grid(True)

# Improve layout and display the plot
plt.tight_layout()
plt.show()

#%% Principal Component Analysis (PCA)
# Selecting numerical columns for PCA
features = ['Price', 'Rating', 'Review Count', 'Age']
X = fashion_data[features]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA without reducing dimensionality too much to examine singular values
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio per component
print("Explained Variance Ratio per Component:", pca.explained_variance_ratio_)

# Cumulative Variance Explained
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print("Cumulative Variance Explained:", cumulative_variance)

# Printing Singular Values
print("Singular Values:", pca.singular_values_)

# Calculating Condition Number
condition_number = pca.singular_values_[0] / pca.singular_values_[-1]
print("Condition Number:", condition_number)
print("""Explained Variance Ratio per Component:
Each component explains a certain proportion of the total variance in the dataset.
For example, the first component explains approximately 25.05% of the total variance, the second explains about 25.03%, and so on.
Cumulative Variance Explained:
Indicates the cumulative proportion of total variance explained by the components.
For instance, the first two components together explain around 50.08% of the total variance, the first three explain about 75.06%, and all four components collectively explain 100%.
Singular Values:
Represent the square roots of the eigenvalues of the covariance matrix.
Higher singular values correspond to components that capture more variance in the data.
Condition Number:
Reflects the sensitivity of the solution of a linear system of equations to changes in the input.
A condition number close to 1 indicates that the system is well-conditioned and less sensitive to changes.""")
#%% PCA Fig 1
# Plotting the explained variance ratio per component
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--', color='b')
plt.title('Explained Variance Ratio per Component')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.grid(True)
plt.show()
#%% PCA Fig 2
# Plotting the cumulative variance explained
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='r')
plt.title('Cumulative Variance Explained')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.grid(True)
plt.show()
#%%Normality Test/Transformation
from scipy.stats import anderson

numerical_columns = ['Price', 'Rating', 'Review Count', 'Age']

# Set up subplots
fig, axs = plt.subplots(len(numerical_columns), 1, figsize=(10, 8), sharex=True)

# Iterate through each numerical column
for i, column in enumerate(numerical_columns):
    # Perform Anderson-Darling test for normality
    result = anderson(fashion_data[column])

    # Plot histogram
    axs[i].hist(fashion_data[column], bins='auto', color='skyblue', alpha=0.7, rwidth=0.85)
    axs[i].set_title(f'Histogram of {column} (Anderson-Darling Statistic: {result.statistic:.2f})')
    axs[i].set_xlabel('Value')
    axs[i].set_ylabel('Frequency')
    axs[i].grid(axis='y', alpha=0.75)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

#%%Transformation
numerical_columns = ['Price', 'Rating', 'Review Count', 'Age']

# Iterate through each numerical column
for column in numerical_columns:
    # Get the data for the current column
    column_data = fashion_data[column]

    # Perform Anderson-Darling test for normality
    statistic, critical_values, significance_levels = anderson(column_data)

    # Print Anderson-Darling test results
    print(f"\nAnderson-Darling Test Results for '{column}' column:")
    print("Statistic:", statistic)
    print("Critical Values:", critical_values)
    print("Significance Levels:", significance_levels)

    # Check if the data is normally distributed based on the test statistic
    if statistic < critical_values[2]:
        print("The data follows a normal distribution (fail to reject H0)")
    else:
        print("The data does not follow a normal distribution (reject H0)")

#%%Pearson Correlation Coefficient Matrix
# Compute the Pearson correlation coefficient matrix for numerical columns
numerical_columns = fashion_data.select_dtypes(include=['float64', 'int64']).columns
numerical_data = fashion_data[numerical_columns]

correlation_matrix = numerical_data.corr()

# Round the values in the correlation matrix to 2 decimal places
correlation_matrix_rounded = correlation_matrix.round(2)

# Create a PrettyTable object
table = PrettyTable()

# Add the column names to the table
table.field_names = correlation_matrix_rounded.columns

# Add rows to the table
for index, row in correlation_matrix_rounded.iterrows():
    table.add_row([f"{value:.2f}" for value in row])

# Print the table
print("Pearson Correlation Coefficient Matrix:")
print(table)
#%%Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", annot_kws={"size": 10})
plt.title("Pearson Correlation Coefficient Heatmap")
plt.show()

#%%Scatter Plot Matrix
sns.pairplot(numerical_data, kind='scatter')
plt.suptitle("Scatter Plot Matrix", y=1.02)
plt.show()

#%%Statistical Analysis-KDE Plots
# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Plot multivariate kernel density estimate for each column
for i, column in enumerate(numerical_columns):
    sns.kdeplot(data=fashion_data[column], ax=axes[i], fill=True)
    axes[i].set_title(column)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Density")

# Adjust layout
plt.tight_layout()
plt.show()

#%%Line Plot
# Set font and size for title
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Group the data by 'Brand' and calculate the median price for each brand
brand_price_median = fashion_data.groupby('Brand')['Price'].median()

# Create a line plot for 'Price' based on 'Brand'
plt.figure(figsize=(10, 6))
plt.plot(brand_price_median.index, brand_price_median.values, marker='o', color='blue')

plt.title("Line Plot of Price based on Brand", color='blue')
plt.xlabel("Brand", color='darkred')
plt.ylabel("Price", color='darkred')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()
plt.show()

#%%Bar Plot-Stacked
# Group the data by category and season, and calculate total sales
sales_data = fashion_data.groupby(['Category', 'Season'])['Price'].sum().unstack()

# Plot stacked bar plot
plt.figure(figsize=(10, 6))
sales_data.plot(kind='bar', stacked=True, cmap='viridis')

plt.title("Total Sales by Category and Season", color='blue')
plt.xlabel("Category", color='darkred')
plt.ylabel("Total Sales", color='darkred')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Barplot-Group
grouped_data = fashion_data.groupby(['Available Sizes', 'Color'])['Price'].count().unstack()

# Plot grouped bar plot
plt.figure(figsize=(10, 6))
grouped_data.plot(kind='bar', cmap='viridis')

plt.title("Price of Products by Sizes and Color", color='blue')
plt.xlabel("Size", color='darkred')
plt.ylabel("Price", color='darkred')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.legend(title='Color',bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

#%%Count plot
# Get the top 5 brands based on maximum price
top_5_brands = fashion_data.groupby('Brand')['Price'].max().nlargest(5).index.tolist()

# Filter the DataFrame to include only rows with the top 5 brands
top_5_brands_data = fashion_data[fashion_data['Brand'].isin(top_5_brands)]

# Set font and size for title
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Create count plot for top 5 brands
plt.figure(figsize=(10, 6))
sns.countplot(data=top_5_brands_data, x='Brand', order=top_5_brands, palette='viridis')

plt.title("Count of Products for Top 5 Brands based on Maximum Price", color='blue')
plt.xlabel("Brand", color='darkred')
plt.ylabel("Count", color='darkred')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()
plt.show()

#%%Pie chart
import math

# Group by 'Fashion Magazines' and 'Purchase History', and count occurrences of each
grouped = fashion_data.groupby(['Fashion Magazines', 'Purchase History']).size().reset_index(name='count')

# Create a pivot table to rearrange the data for plotting
pivot_table = grouped.pivot(index='Fashion Magazines', columns='Purchase History', values='count')

# Calculate percentage of each 'Purchase History' category for each 'Fashion Magazines'
pivot_table_percentage = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

# Determine the number of rows and columns for subplots
num_magazines = len(pivot_table_percentage.index)
num_rows = 5 if num_magazines > 5 else num_magazines  # Number of rows for subplots (max 5)
num_cols = math.ceil(num_magazines / num_rows)  # Calculate number of columns based on number of magazines

# Create subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 8))

# Flatten axes if necessary
if num_magazines == 1:
    axes = [axes]

# Plot pie charts for each 'Fashion Magazines'
for idx, magazine in enumerate(pivot_table_percentage.index):
    ax = axes[idx // num_cols, idx % num_cols]
    patches, texts, autotexts = ax.pie(pivot_table_percentage.loc[magazine], autopct='%1.1f%%', startangle=140)
    ax.set_title(f'{magazine[:15]}...')  # Shorten title
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Adjust fontsize of percentage labels
    for autotext in autotexts:
        autotext.set_fontsize(6)

# Create a legend outside of the loop
fig.legend(pivot_table_percentage.columns, loc='upper right', bbox_to_anchor=(1, 1))

# Adjust layout
plt.tight_layout()

# Show the subplots
plt.show()

#%%Dist plot
# Filter the data for the desired columns
price_season_data = fashion_data[['Price', 'Season']]

# Create the combined plot
plt.figure(figsize=(10, 6))

# Plot the distribution of 'Price' for each season with a kernel density estimate line
sns.kdeplot(data=price_season_data[price_season_data['Season'] == 'Fall/Winter']['Price'], label='Fall/Winter', color='blue')
sns.kdeplot(data=price_season_data[price_season_data['Season'] == 'Spring/Summer']['Price'], label='Spring/Summer', color='orange')
sns.kdeplot(data=price_season_data[price_season_data['Season'] == 'Winter']['Price'], label='Winter', color='green')

# Set labels and title
plt.xlabel('Price')
plt.ylabel('Density')
plt.title('Distribution of Price by Season')

# Show the legend
plt.legend()

# Show the plot
plt.show()
#%%Pair plot
# Select numerical columns for pair plot
numerical_columns = ['Price', 'Age', 'Review Count', 'Rating']

# Create pair plot
sns.pairplot(fashion_data[numerical_columns])

# Display the pair plot
plt.title('Pair Plot of Numerical Columns')
plt.show()
#%%Heat Map with Cbar
# Select numerical columns for correlation analysis
numerical_columns = ['Price', 'Age', 'Review Count', 'Rating']

# Calculate the correlation matrix
correlation_matrix = fashion_data[numerical_columns].corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cbar=True)

# Set title and display
plt.title('Correlation Heatmap of Numerical Columns')
plt.show()

#%%Histogram with KDE
# Create histogram plot with KDE for 'Age' column in the dataset
plt.figure(figsize=(8, 6))
sns.histplot(fashion_data['Age'], kde=True, color='blue', bins=30)

# Set labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution with KDE')

# Show plot
plt.show()

#%%QQ Plot

import statsmodels.api as sm
import matplotlib.pyplot as plt

# Extract the numerical column you want to analyze (e.g., 'Price')
numerical_column = 'Price'

# Create the QQ plot
plt.figure(figsize=(8, 6))
sm.qqplot(fashion_data[numerical_column], line='s', marker='o', markersize=4, alpha=0.5)
plt.title('QQ Plot for Price')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)
plt.show()

#%%KDE Plot
# Select the column you want to plot (e.g., 'Age')
selected_column = 'Rating'

# Set the palette and linewidth
palette = 'viridis'  # You can choose any palette available in seaborn
linewidth = 2

# Create the KDE plot with filled areas
plt.figure(figsize=(8, 6))
sns.kdeplot(data=fashion_data[selected_column], fill=True, alpha=0.6, palette=palette, linewidth=linewidth)
plt.title(f'KDE Plot for {selected_column}')
plt.xlabel(selected_column)
plt.ylabel('Density')
plt.grid(True)
plt.show()

#%%Im or reg plot with scatter representation and regression line
# Select the columns you want to plot (e.g., 'Age' and 'Price')
x_column = 'Rating'
y_column = 'Review Count'

# Create the scatter plot with regression line
plt.figure(figsize=(8, 6))
sns.lmplot(x=x_column, y=y_column, data=fashion_data, scatter=True, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5})
plt.title(f'Scatter Plot with Regression Line')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.grid(True)
plt.tight_layout()
plt.show()

#%%Multivariate Box or Boxen plot
# Select the columns you want to include in the plot (e.g., 'Age' and 'Price')
columns_to_plot = ['Age', 'Price']

# Create a multivariate box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=fashion_data[columns_to_plot])
plt.title('Multivariate Box Plot')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.show()

# Create a multivariate boxen plot
plt.figure(figsize=(10, 6))
sns.boxenplot(data=fashion_data[columns_to_plot])
plt.title('Multivariate Boxen Plot')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.show()

#%%Area plot
agg_data = fashion_data.groupby(['Brand', 'feedback']).agg({'Price': 'mean'}).reset_index()

plt.figure(figsize=(10, 6))
for season, season_data in agg_data.groupby('feedback'):
    sns.lineplot(data=season_data, x='Brand', y='Price', label=season, lw=2, alpha=0.6)
plt.title('Mean Price by Brand with feedback Variation')
plt.xlabel('Brand')
plt.ylabel('Mean Price')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title='feedback', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.tight_layout()
plt.show()

#%%Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='Season', y='Price', data=fashion_data)
plt.title('Violin Plot of Price by Season')
plt.xlabel('Season')
plt.ylabel('Price')
plt.show()

#%%Joint plot with KDE and scatter representation
from sklearn.preprocessing import LabelEncoder
# Convert categorical variables to numerical using label encoding
label_encoder = LabelEncoder()
fashion_data['Fashion Magazines'] = label_encoder.fit_transform(fashion_data['Fashion Magazines'])
fashion_data['Fashion Influencers'] = label_encoder.fit_transform(fashion_data['Fashion Influencers'])

# Find correlation between the two numerical variables
correlation = fashion_data['Fashion Magazines'].corr(fashion_data['Fashion Influencers'])

# Create joint plot with KDE and scatter representation
joint_plot = sns.jointplot(x='Fashion Magazines', y='Fashion Influencers', data=fashion_data, kind='scatter', height=7)
joint_plot.plot_joint(sns.kdeplot, color="blue", zorder=0, levels=6)  # Add KDE plot

# Set plot title and labels
plt.xlabel('Fashion Magazines')
plt.ylabel('Fashion Influencers')
plt.title(f'Joint Plot with KDE and Scatter Representation\nCorrelation: {correlation:.2f}')

# Show the plot
plt.tight_layout()
plt.show()

#%%Rug plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=fashion_data, x='Customer Reviews', y='Social Media Comments', hue='Season', alpha=0.5)
sns.rugplot(data=fashion_data, x='Customer Reviews', height=0.1, color='blue', alpha=0.5)
sns.rugplot(data=fashion_data, y='Social Media Comments', height=0.1, color='red', alpha=0.5)
plt.title('Rug Map of Social Media Comments vs Customer Reviews')
plt.xlabel('Customer Reviews')
plt.ylabel('Social Media Comments')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%% 3D Scatter plot

from mpl_toolkits.mplot3d import Axes3D
# Randomly sample 1000 rows from the dataframe
sampled_data = fashion_data.sample(n=1000, random_state=42)

# Extracting the columns for the 3D scatter plot
price = sampled_data['Price']
rating = sampled_data['Rating']
review_count = sampled_data['Review Count']

# Creating a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(price, rating, review_count, c='blue', marker='o')

# Setting labels and title
ax.set_xlabel('Price')
ax.set_ylabel('Rating')
ax.set_zlabel('Review Count')
ax.set_title('3D Scatter Plot of Price, Rating, and Review Count')

plt.show()

#%%Contour plot
# Contour plot

sampled_data = fashion_data.sample(n=1000, random_state=42)

# Extracting the columns for the contour plot
price = sampled_data['Price']
rating = sampled_data['Rating']

# Creating the contour plot
plt.figure(figsize=(8, 6))
sns.kdeplot(x=price, y=rating, cmap='viridis', fill=True)
plt.xlabel('Price')
plt.ylabel('Rating')
plt.title('Contour Plot of Price and Rating')

plt.show()

#%%Cluster map
numerical_columns = fashion_data.select_dtypes(include=['float64', 'int64'])

# Sample the numerical data
sample_data = numerical_columns.sample(n=100)  # Adjust the number of samples as needed

# Create a cluster map
sns.clustermap(sample_data.corr(), cmap='coolwarm', linewidths=0.5, figsize=(10, 10))
plt.title('Cluster Map of Numerical Columns')
plt.tight_layout()
plt.show()

#%%Hexbin
# Taking a random sample of 1000 rows from the dataframe
sampled_data = fashion_data.sample(n=1000)

# Encoding categorical columns to numerical codes
sampled_data['Season_Code'] = pd.factorize(sampled_data['Season'])[0]
sampled_data['Brand_Code'] = pd.factorize(sampled_data['Brand'])[0]

# Creating the hexbin plot
plt.figure(figsize=(10, 6))
plt.hexbin(sampled_data['Brand_Code'], sampled_data['Season_Code'], gridsize=30, cmap='viridis')
plt.colorbar(label='count in bin')
plt.xlabel('Brand')
plt.ylabel('Season')
plt.title('Hexbin Plot of Brand vs Season')
plt.xticks(ticks=range(len(sampled_data['Brand'].unique())), labels=sampled_data['Brand'].unique(), rotation=45)
plt.tight_layout()
plt.show()

#%%Strip plot
import random

# Assuming 'fashion_data' is your DataFrame
sample_size = 1000  # Adjust the sample size as needed
random_magazines = random.sample(fashion_data['Fashion Magazines'].unique().tolist(), 5)  # Select random Fashion Magazines

# Randomly sample from the DataFrame
fashion_data_sample = fashion_data.sample(n=sample_size, random_state=42)

plt.figure(figsize=(10, 6))
sns.stripplot(x='Fashion Influencers', y='Price', data=fashion_data_sample, jitter=True, hue='Fashion Magazines', marker='o', alpha=0.5)
plt.title('Strip Plot of Price by Fashion Influencers')
plt.xlabel('Fashion Influencers')
plt.ylabel('Price')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title='Fashion Magazines', labels=random_magazines,bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()# Set random Fashion Magazines as legend labels
plt.show()

#%%Swarm plot
# Take a random sample of 1000 rows from the DataFrame
sample_data = fashion_data.sample(n=1000, random_state=42)

plt.figure(figsize=(10, 6))
sns.swarmplot(x='Rating', y='Total Sizes', data=sample_data, hue='Available Sizes')
plt.title('Swarm Plot of Rating by Total Sizes')
plt.xlabel('Rating')
plt.ylabel('Total Sizes')
plt.legend(title='Available Sizes',bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

