
%7. Find the nearest cluster for test data 
%to find the distance of test data from each training data point, we can
%actually find the distance of test data from the centroid of each cluster
% using Euclidean distance a sthe metrics

%take the input test file
testMealData=csvread('mealDataX.csv',0,0);

%find the feature matrix for test data 
test_matrix= find_features(testMealData);

% read the centroids from kmeans
centroidK =csvread('centroid_kmeans.csv',0,0);

% find the closest clusters for each test data

[~,idx_test] = pdist2(centroidK ,test_matrix,'euclidean','Smallest',1);

% read the centroids from ground truth clusters
centroid_clusters =csvread('centroid_clusters.csv',0,0);

% find the closest clusters for each test data

[~,idx_test_c] = pdist2(centroid_clusters ,test_matrix,'euclidean','Smallest',1);
