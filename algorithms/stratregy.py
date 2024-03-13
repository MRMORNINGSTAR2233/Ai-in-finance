# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to perform K-Means clustering on income data
def income_clustering(income_data, num_clusters):
    # Assuming income_data is a DataFrame with a single column 'Income'
    X = income_data[['Income']].values
    
    # Initialize K-Means model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    
    # Fit the model to the data
    kmeans.fit(X)
    
    # Add cluster labels to the original DataFrame
    income_data['Cluster'] = kmeans.labels_
    
    return income_data, kmeans

# Function to visualize the clustering results
def plot_clusters(income_data, kmeans_model):
    plt.scatter(income_data['Income'], [0] * len(income_data), c=income_data['Cluster'], cmap='viridis')
    plt.scatter(kmeans_model.cluster_centers_[:, 0], [0] * len(kmeans_model.cluster_centers_), marker='X', s=200, c='red')
    plt.title('Income Clusters')
    plt.xlabel('Income')
    plt.show()

# Function to suggest diversification strategies based on income clusters
def diversification_strategies(income_data):
    # Calculate mean income for each cluster
    cluster_means = income_data.groupby('Cluster')['Income'].mean().sort_values()
    
    # Identify the cluster with the lowest mean income
    lowest_income_cluster = cluster_means.idxmin()
    
    # Identify the cluster with the highest mean income
    highest_income_cluster = cluster_means.idxmax()
    
    # Suggest diversification strategy
    diversification_strategy = f"Suggested diversification strategy: Allocate resources from Cluster {highest_income_cluster} to Cluster {lowest_income_cluster}."
    
    return diversification_strategy

# Example usage
if __name__ == "__main__":
    # Sample income data (replace this with your actual income data)
    income_data = pd.DataFrame({'Income': [50000, 60000, 70000, 80000, 120000, 130000, 140000, 150000]})
    
    # Set the number of clusters
    num_clusters = 2
    
    # Perform K-Means clustering
    clustered_income_data, kmeans_model = income_clustering(income_data, num_clusters)
    
    # Visualize the clustering results
    plot_clusters(clustered_income_data, kmeans_model)
    
    # Suggest diversification strategies
    strategy = diversification_strategies(clustered_income_data)
    
    print(strategy)
