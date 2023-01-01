#loading libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score               
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture
from mpl_toolkits import mplot3d

#Function to display data distribution
def showDist(df):
    
    #Underlying data distribution
    fig, axs = plt.subplots(3, 2)
    ax1 = sns.kdeplot(ax = axs[0,0], data = df[['Temp.Min', 'Temp.Max', 'Temp.Mean', 'Humidity.Min', 'Humidity.Max', 'Humidity.Mean']], fill=True)
    ax2 = sns.kdeplot(ax = axs[0,1], data = df[['Windgust.Min', 'Windgust.Max', 'Windgust.Mean', 'Windspeed.Min', 'Windspeed.Max', 'Windspeed.Mean']],  fill=True)
    ax3 = sns.kdeplot(ax = axs[1,0], data = df[['Pressure.Min', 'Pressure.Max', 'Pressure.Mean']],fill=True)
    ax4 = sns.kdeplot(ax = axs[1,1], data = df[['Precipitation',]], fill=True)
    ax5 = sns.kdeplot(ax = axs[2,0], data = df['Snowfall.Amount'], fill=True)
    ax6 = sns.kdeplot(ax = axs[2,1], data = df['Sunshine.Duration' ],fill=True)
    plt.show()
    
#Function to display boxplots for inout features
def showBoxplot(df):
    
    #Displaying the boxplots of the features
    
    plt.figure(figsize=(18,12))
    plt.boxplot(df[['Temp.Min', 'Temp.Max', 'Temp.Mean', 'Humidity.Min','Humidity.Max', 'Humidity.Mean', 'Precipitation', 'Snowfall.Amount', 'Windgust.Min',
                    'Windgust.Max', 'Windgust.Mean', 'Windspeed.Min', 'Windspeed.Max', 'Windspeed.Mean']])
    plt.xticks(range(1,15), ['Temp.Min', 'Temp.Max', 'Temp.Mean', 'Humidity.Min','Humidity.Max', 'Humidity.Mean', 'Precipitation', 'Snowfall.Amount',
                'Windgust.Min', 'Windgust.Max', 'Windgust.Mean', 'Windspeed.Min', 'Windspeed.Max', 'Windspeed.Mean'], rotation = 45)
    plt.show()

    plt.boxplot(df[['Pressure.Min', 'Pressure.Max', 'Pressure.Mean']])
    plt.xticks(range(1,4), ['Pressure.Min', 'Pressure.Max', 'Pressure.Mean'], rotation =45)
    plt.show()
    
    plt.boxplot(df['Sunshine.Duration'])
    plt.xticks([1], ['Sunshine.Duration'])
    plt.show()

#Function to run K-Means algorithm
def kmeansclustering(climateNormalisedDF, climateColumns):
    
    #################################
    ###### HANDLING OUTLIERS ########
    #################################


    originalRows = climateNormalisedDF.shape[0]
    
    for column in climateColumns:
        climateNormalisedDF = climateNormalisedDF[(climateNormalisedDF[column] < 3) & (climateNormalisedDF[column] > -3)]
    
    newRows = climateNormalisedDF.shape[0]
    
    #Fetching a list of index RETAINED from the original dataframe post removing outliers
    index = climateNormalisedDF.index

    #Percentage of rows dropped post removing outliers
    print('Percentage of outliers dropped ', round((((originalRows-newRows)/originalRows)*100),2), '%')

    ########################################
    ###### DIMENSIONALITY REDUCTION ########
    ########################################

    #Running PCA on our dataset
    pca = PCA(n_components=4)
    climatePCAArray = pca.fit_transform(climateNormalisedDF.to_numpy())
    print('Variance explained by each component - ', pca.explained_variance_ratio_)
    print('Variance explained by 4 components ', round((sum(pca.explained_variance_ratio_)*100),2), '%' )
    print('Co-effecient of original component to the PCA axis - ', pca.components_)

    #Variance explained by each component of the PCA graph
    plt.figure(figsize=(6,4))
    plt.bar(range(4), pca.explained_variance_ratio_,tick_label=range(4))
    plt.step(range(4), pca.explained_variance_ratio_.cumsum())
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained (%)")
    plt.show()
    

    #Plotting PCA axis 1 and axis 2 which explains maximum variance.
    plt.figure(figsize=(6,4))
    plt.plot(climatePCAArray[:,0],climatePCAArray[:,1],".")
    plt.xlabel("1st Principal Component")
    plt.ylabel("2nd Principal Component")
    plt.show()

    #Plotting a heatmap for the relation between original variables and the transformed PCA Axis
    map = pd.DataFrame(pca.components_,columns=climateColumns)
    plt.figure(figsize=(18,6))
    sns.heatmap(map,cmap='rocket', annot=True)
    plt.show()

    ##################################################
    ###### DETERMINING THE NUMBER OF CLUSTERS ########
    ##################################################

    #Using Elbow method to determine the number of clusters
    wcss = []
    for i in range(1, 20):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(climatePCAArray)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 20), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    #Using Silhoeutte method for determining the number of clusters
    n_cluster = range(2,20)
    sil_avg = []
    
    for n in n_cluster:
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(climatePCAArray)
        cluster_labels = kmeans.labels_
        sil_avg.append(silhouette_score(climatePCAArray, cluster_labels))
    
    plt.plot(n_cluster,sil_avg,'bx-')
    plt.xlabel('Value of clusters')
    plt.ylabel('Silhouetter score')
    plt.title('Silhouette analysis For Optimal k')
    plt.show()

    #Finding the number of clusters from elbow method
    kn = KneeLocator(x=range(1,20), 
                    y=wcss, 
                    curve='convex', 
                    direction='decreasing')

    print('The number of clusters to be formed based on knee locator is ', kn.knee)

    ##############################
    ###### MODELLING DATA ########
    ##############################

    # We have chosen five clusters, so we run K-means with number of clusters equals four.
    # Same initializer and random state as before.
    kmeans = KMeans(n_clusters = kn.knee, init = 'k-means++', random_state = 42, max_iter=300)
    clusteredData = kmeans.fit_predict(climatePCAArray)
    print('Inertia of the k-mean algorith with ',kn.knee,' clusters ', kmeans.inertia_)

    #################################
    ###### PLOTTING CLUSTERS ########
    #################################

    # We create a new data frame with the original features and add the PC scores and assigned clusters.
    climateClusteredDF = pd.concat([climateNormalisedDF.reset_index(drop = True), pd.DataFrame(climatePCAArray)], axis = 1)
    climateClusteredDF.columns.values[-4: ] = ['Component 1', 'Component 2', 'Component 3', 'Component 4']
    climateClusteredDF['clusteredData'] = clusteredData

    #Plotting the CLusters against first two PCA axis
    plt.figure(figsize=(18,12))
    plt.scatter(climateClusteredDF.loc[:, 'Component 1'], climateClusteredDF.loc[:, 'Component 2'], c = climateClusteredDF.loc[:, 'clusteredData'], cmap='viridis')
    plt.xlabel("1st Principal Component")
    plt.ylabel("2nd Principal Component")
    plt.show()

    #Plotting the clusters for the first 3 principal component
    plt.figure(figsize=(18,12))
    ax = plt.axes(projection='3d')

    # Data for three-dimensional scattered points
    zdata = climateClusteredDF.loc[:, 'Component 3']
    xdata = climateClusteredDF.loc[:, 'Component 1']
    ydata = climateClusteredDF.loc[:, 'Component 2']
    ax.scatter3D(xdata, ydata, zdata, c=climateClusteredDF['clusteredData'], cmap='rainbow')
    ax.legend()
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.show()

    return clusteredData, index

#Function for gaussian clustering
def gaussianclustering(climateNormalisedDF, climateColumns):

    ########################################
    ###### DIMENSIONALITY REDUCTION ########
    ########################################

    #Running PCA on our dataset
    pca = PCA(n_components=5)
    climatePCAArray = pca.fit_transform(climateNormalisedDF.to_numpy())
    climatePCADF = pd.DataFrame(climatePCAArray)

    print('Variance explained by each component - ', pca.explained_variance_ratio_)
    print('Variance explained by 5 components ', round((sum(pca.explained_variance_ratio_)*100),2), '%' )
    print('Co-effecient of original component to the PCA axis - ', pca.components_)

    #Variance explained by each component of the PCA graph
    plt.figure(figsize=(6,4))
    plt.bar(range(5), pca.explained_variance_ratio_,tick_label=range(5))
    plt.step(range(5), pca.explained_variance_ratio_.cumsum())
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained (%)")
    plt.show()

    #Plotting PCA axis 1 and axis 2 which explains maximum variance.
    plt.figure(figsize=(6,4))
    plt.plot(climatePCAArray[:,0],climatePCAArray[:,1],".")
    plt.xlabel("1st Principal Component")
    plt.ylabel("2nd Principal Component")
    plt.show()

    #Plotting a heatmap for the relation between original variables and the transformed PCA Axis
    map = pd.DataFrame(pca.components_,columns=climateColumns)
    plt.figure(figsize=(18,6))
    sns.heatmap(map,cmap='rocket', annot=True)
    plt.show()

    ##################################################
    ###### DETERMINING THE NUMBER OF CLUSTERS ########
    ##################################################

    #Selecting best GMM model using BIC score
    bic = []
    n_clusters = np.arange(3,20)
    for i in n_clusters:

        gmm = GaussianMixture(n_components=i, random_state=42)
        gmm.fit(climatePCADF)
        bic.append(gmm.bic(climatePCADF))

    #Plotting BIC score
    plt.plot(n_clusters, bic)
    plt.xticks(n_clusters)
    plt.title('BIC score for different clusters')
    plt.xlabel("No. of clusters")
    plt.ylabel("BIC score")
    plt.show()

    #Gradient of the BIC
    plt.errorbar(n_clusters, np.gradient(bic), label='BIC')
    plt.title("Gradient of BIC Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("No. of clusters")
    plt.ylabel("Gradient of BIC)")
    plt.legend()
    plt.show()

    ##############################
    ###### MODELLING DATA ########
    ##############################

    #Running GMM model with 5 clusters
    gmm = GaussianMixture(n_components=5, random_state=42)
    gmm.fit(climatePCADF)
    # predict the cluster for each data point
    y_cluster_gmm = gmm.predict(climatePCADF)

    #Means of the underlying distribution
    print(gmm.means_)
    #Covariance of the underlying distribution
    print(gmm.covariances_)

    #################################
    ###### PLOTTING CLUSTERS ########
    #################################

    # We create a new data frame with the original features and add the PC scores and assigned clusters.
    climateClusteredDF = pd.concat([climateNormalisedDF.reset_index(drop = True), pd.DataFrame(climatePCAArray)], axis = 1)
    climateClusteredDF.columns.values[-5: ] = ['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5']
    climateClusteredDF['clusteredData'] = y_cluster_gmm

    #Plotting the CLusters against first two PCA axis
    plt.figure(figsize=(18,12))
    plt.scatter(climateClusteredDF.loc[:, 'Component 1'], climateClusteredDF.loc[:, 'Component 2'], c = climateClusteredDF.loc[:, 'clusteredData'], cmap='viridis')
    plt.xlabel("1st Principal Component")
    plt.ylabel("2nd Principal Component")
    plt.show()

    #Plotting the clusters for the first 3 principal component
    plt.figure(figsize=(18,12))
    ax = plt.axes(projection='3d')

    # Data for three-dimensional scattered points
    zdata = climateClusteredDF.loc[:, 'Component 3']
    xdata = climateClusteredDF.loc[:, 'Component 1']
    ydata = climateClusteredDF.loc[:, 'Component 2']
    ax.scatter3D(xdata, ydata, zdata, c=climateClusteredDF['clusteredData'], cmap='rainbow')
    ax.legend()
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.show()
    
    return y_cluster_gmm

#Interpreting the clusters of K-means
def interpretKMeans(climateKMeansDF, clusterData, index):
    
    #Filtering outliers from the original dataset, based on the one's calculated from the z-score.
    climateNoOutlierDF = climateKMeansDF.iloc[index]
    climateNoOutlierDF = climateNoOutlierDF.reset_index(drop=True)
    climateNoOutlierDF['clusterData'] = clusterData
    
    #################################################
    ###### DATA DISTRIBUTION BY EACH CLUSTER ########
    #################################################
    
    #We are going to interpret clusters based on Mean value of correlated features and the other independent features
    climateAnalysisDF = climateNoOutlierDF[['Temp.Mean', 'Pressure.Mean', 'Humidity.Mean', 'Precipitation', 'Windspeed.Mean', 'Windgust.Mean', 'Sunshine.Duration', 'Snowfall.Amount', 'clusterData']]
    
    clusterNumbers = climateNoOutlierDF['clusterData'].unique()
    clusterNumbers.sort()
    dataDistribution = {'Dist':[]}


    for cluster in clusterNumbers:
        #Extracting a particular column and cluster data
        dataDistTemp = climateAnalysisDF
        #Removing that particular cluster rows.
        dataDistTemp = dataDistTemp[dataDistTemp['clusterData'] == cluster]
        distribution = len(dataDistTemp)/len(climateAnalysisDF)*100
        dataDistribution['Dist'].append(distribution)

    print('Distribution of cluster Data points : ',dataDistribution)
    
    ###############################################################
    ###### PLOTTING FEATURE DATA DISTRIBUTION FOR CLUSTERS ########
    ###############################################################

    for c in climateAnalysisDF:
        g = sns.FacetGrid(climateAnalysisDF, col= 'clusterData')
        g.map(plt.hist, c)
    plt.show()
    
    
#Interpreting the clusters of GMM
def interpretGMM(climateGMMDF, clusterData):
    
    #Appending assigned cluster number to data frame
    climateGMMDF['clusterData'] = clusterData
    
    #We are going to interpret clusters based on Mean value of correlated features and the other independent features
    climateAnalysisDF = climateGMMDF[['Temp.Mean', 'Pressure.Mean', 'Humidity.Mean', 'Precipitation', 'Windspeed.Mean', 'Windgust.Mean', 'Sunshine.Duration', 'Snowfall.Amount', 'clusterData']]
    
    #Number of clusters
    clusterNumbers = climateGMMDF['clusterData'].unique()
    clusterNumbers.sort()
    
    #################################################
    ###### DATA DISTRIBUTION BY EACH CLUSTER ########
    #################################################

    #Creating a dictionary to hold the distribution of data under each distribution
    dataDistribution = {'Dist':[]}

    for cluster in clusterNumbers:
        #Extracting a particular column and cluster data
        dataDistTemp = climateAnalysisDF
        #Removing that particular cluster rows.
        dataDistTemp = dataDistTemp[dataDistTemp['clusterData'] == cluster]
        distribution = len(dataDistTemp)/len(climateAnalysisDF)*100
        dataDistribution['Dist'].append(distribution)

    print('Distribution of Data points under each gaussian : ',dataDistribution)
    
    ###############################################################
    ###### PLOTTING FEATURE DATA DISTRIBUTION FOR CLUSTERS ########
    ###############################################################

    for c in climateAnalysisDF:
        g = sns.FacetGrid(climateAnalysisDF, col= 'clusterData')
        g.map(plt.hist, c)
    plt.show()



def main():
    
    # Creating a dataframe and numpy matrix of the data
    climateColumns=['Temp.Min', 'Temp.Max', 'Temp.Mean', 'Humidity.Min', 'Humidity.Max', 'Humidity.Mean',
            'Pressure.Min', 'Pressure.Max', 'Pressure.Mean', 'Precipitation', 'Snowfall.Amount', 'Sunshine.Duration',
            'Windgust.Min', 'Windgust.Max', 'Windgust.Mean', 'Windspeed.Min', 'Windspeed.Max', 'Windspeed.Mean']
    climateDF = pd.read_csv("/Users/AbhishekGupta/Documents/Lancaster/DataMining/CW/CW1/ClimateDataBasel.csv", names = climateColumns, header = None, index_col=False)


    ###########################
    ###### DATA ANALYSIS ######
    ###########################

    #Describe the Data Frame
    print(climateDF.describe().transpose())

    #Underlying data distribution
    showDist(climateDF)

    #Checking for correlation between the dataset
    plt.figure(figsize=(18,12))
    sns.heatmap(climateDF.corr(), annot = True,vmin=-1, vmax=1)
    plt.show() 
    #Checking for Null values
    climateDF.isnull().values.any() 

    #Plotting a boxplot for thr input variables
    #The boxplots have been grouped for categories with similar range of values.
    showBoxplot(climateDF)

    #################################
    ###### DATA PRE-PROCESSING ######
    #################################

    #satndardizing the data
    scaler = StandardScaler() 
    climateNormalisedArray = scaler.fit_transform(climateDF)
    climateNormalisedDF = pd.DataFrame(climateNormalisedArray, columns = climateColumns)
    
    ############################
    ###### DATA MODELLING ######
    ############################

    #Calling the K-Means algorithm
    clusterData, index = kmeansclustering(climateNormalisedDF, climateColumns)
    #Creating a copy of original dataframe for K-Means Interpretation
    climateKMeansDF = climateDF.copy()
    #Creating graphs to interpret k-means clusters
    interpretKMeans(climateKMeansDF, clusterData, index)

    #Calling gaussian mixture model
    clusterData = gaussianclustering(climateNormalisedDF, climateColumns)
    #Creating a copy of original dataframe for GMM Interpretation
    climateGMMDF = climateDF.copy()
    #Interpreting GMM clusters
    interpretGMM(climateGMMDF, clusterData)



if __name__ == '__main__':
    main()