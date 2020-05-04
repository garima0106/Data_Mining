M1=csvread('mealData1.csv',0,0);
M2=csvread('mealData2.csv',0,0);
M3=csvread('mealData3.csv',0,0);
M4=csvread('mealData4.csv',0,0);
M5=csvread('mealData5.csv',0,0);


MealData= [M1;M2;M3;M4;M5];


%load the Carb amount data
C1=csvread('mealAmountData1.csv',0,0);
C2=csvread('mealAmountData2.csv',0,0);
C3=csvread('mealAmountData3.csv',0,0);
C4=csvread('mealAmountData4.csv',0,0);
C5=csvread('mealAmountData5.csv',0,0);


% remove the NaNs and zeros and remove the corresponding row from Meal
% amount data
MealData_train= MealData(:,1:30);

%change all NaNs to zero
MealData_train(isnan(MealData_train))=0;

%find the number of rows which should be deleted
rows_to_be_deleted=0;
delete_rows=[]
for i=1:length(MealData_train(:,1))
    zero_data=0;
    for j=1:length(MealData_train(i,:))
        if MealData_train(i,j)==0 
            zero_data=zero_data+1;
        end
    end
    if zero_data>=1
       rows_to_be_deleted=rows_to_be_deleted+1;
       delete_rows= [delete_rows,i];
    end  
end

% take the first 225 rows in Carb amount data
CarbAmount= [C1;C2];
CarbAmount_New= CarbAmount(1:255,:);

%delete the same rows from carb amount data

Carb_del=CarbAmount_New(:,:);

for i=1:length(delete_rows)
        Carb_del(delete_rows(:,i),:)= -1;
        
end
 
Carb_del(Carb_del==-1) =NaN;

%remove NaN from the data
Carb_del= rmmissing(Carb_del);

%alternate - just to validate 
CarbAmountNew = [CarbAmount_New(1:20,:);CarbAmount_New(23:27,:);CarbAmount_New(34:52,:);CarbAmount_New(54:55,:);CarbAmount_New(60:64,:);CarbAmount_New(66,:);CarbAmount_New(68:72,:);CarbAmount_New(74:79,:);CarbAmount_New(81:85,:);CarbAmount_New(88:90,:);CarbAmount_New(92,:);CarbAmount_New(95:108,:);CarbAmount_New(110:117,:);CarbAmount_New(120:121,:);CarbAmount_New(123:143,:);CarbAmount_New(146:148,:);CarbAmount_New(151:153,:);CarbAmount_New(157:204,:);CarbAmount_New(206:207,:);CarbAmount_New(209:213,:);CarbAmount_New(215:226,:);CarbAmount_New(230:255,:)];

% remove rows from MealData
%set all zeros to NaN
MealData_train(MealData_train==0) =NaN;

%remove NaN from the data
MealData_train= rmmissing(MealData_train);


% 1. Create the bins from ground truth and assign the meal data to corresponding bins
0,% >0 to 20, 21 to 40, 41 to 60, 61 to 80, 81 to 100. 

groundbins=[];
bin1=[]; bin2=[]; bin3=[]; bin4=[]; bin5=[]; bin6=[];
for i= 1:length(MealData_train)
   if CarbAmountNew (i,:) ==0
       bin1 = [bin1,i] ;
       groundbins(i,1)=1;
       
   elseif CarbAmountNew(i,:) > 0 && CarbAmountNew(i,:)<=20
            bin2 = [bin2,i];
            groundbins(i,1)=2;
        elseif CarbAmountNew(i,:) > 20 && CarbAmountNew(i,:)<=40
                bin3 = [bin3,i];
                groundbins(i,1)=3;
        elseif CarbAmountNew(i,:) > 40 && CarbAmountNew(i,:)<=60
                bin4 = [bin4,i];
                groundbins(i,1)=4;
        elseif CarbAmountNew(i,:) > 60 && CarbAmountNew(i,:)<=80
                bin5 = [bin5,i];
                groundbins(i,1)=5;
        elseif CarbAmountNew(i,:) > 80 && CarbAmountNew(i,:)<=100 
                bin6 = [bin6,i];
                groundbins(i,1)=6;
            
            end
            end
            
% 2. Extract features of Meal data

%use the function to find features used in first project
feature_matrix = find_features(MealData);

%feature_matrix_train= feature_matrix;
% feature_matrix is the feature matrix for MealData

% 3.find clusters using kMeans 
% to create - 6 clusters
% each row in idx vector corresponds to the row number in the
% feature_matrix and the value gives the cluster number

[idx,C]=kmeans(feature_matrix,6);

% to find sse, add the third parameter sse
[idx,c,sse] =kmeans(feature_matrix,6);

% plot the cluster centroids and sse values
plot(feature_matrix(idx==1,1),feature_matrix(idx==1,2),'r.','MarkerSize',12)
hold on
plot(feature_matrix(idx==2,1),feature_matrix(idx==2,2),'b.','MarkerSize',12)
plot(feature_matrix(idx==3,1),feature_matrix(idx==3,2),'g.','MarkerSize',12)
plot(feature_matrix(idx==4,1),feature_matrix(idx==4,2),'y.','MarkerSize',12)
plot(feature_matrix(idx==5,1),feature_matrix(idx==5,2),'k.','MarkerSize',12)
plot(feature_matrix(idx==6,1),feature_matrix(idx==6,2),'c.','MarkerSize',12)
plot(c(:,1),'kx','MarkerSize',12,'LineWidth',2)
plot(c(:,2),'ko','MarkerSize',12,'LineWidth',2)
plot(c(:,3),c(:,4),'ko','MarkerSize',12,'LineWidth',2)
plot(c(:,5),c(:,6),'ko','MarkerSize',12,'LineWidth',2)
legend(['Cluster 1 (SSE: ' num2str(sse(1))],...
       ['Cluster 2 (SSE: ' num2str(sse(2))],...
       ['Cluster 3 (SSE: ' num2str(sse(3))],...
       ['Cluster 4 (SSE: ' num2str(sse(4))],...
       ['Cluster 5 (SSE: ' num2str(sse(5))],...
       ['Cluster 6 (SSE: ' num2str(sse(6))],...
   'Centroids', 'Location','NW')


%evaluate clusters based on a criteria

evaS = evalclusters(feature_matrix,idx,'silhouette')

% evaS = 
% 
%   SilhouetteEvaluation with properties:
% 
%     NumObservations: 216
%          InspectedK: 6
%     CriterionValues: 0.2143
%            OptimalK: 6


eva = evalclusters(feature_matrix,idx,'CalinskiHarabasz')

%  eva = 
% 
%   CalinskiHarabaszEvaluation with properties:
% 
%     NumObservations: 216
%          InspectedK: 6
%     CriterionValues: 24.6486
%            OptimalK: 6



% 4. to find supervised validation, compare the results with the bins formed

%function to find total matches between the kmeans results and bins 

match=0;
for i = 1 : length(groundbins(:,1))   
   if groundbins(i,1)==idx(i,1)
    match= match + 1;
    end
end

% match=51
l= length(groundbins)-sum(groundbins(:,1)==0)
%supervised cluster validity metrics
%Classification Error 

Classification_Error_kMeans = match/l

% Classification_Error_kMeans =
% 
%     0.2476


% 5. Use DBSCAN to find the clusters
dbcluster = dbscan(feature_matrix,4,3);

%results in 8 clusters

length(dbcluster(:,1))                 

match_db=0;
for i = 1 : length(groundbins(:,1))   
   if groundbins(i,1)==dbcluster(i,1)
    match_db= match_db + 1;
    end
end

%supervised cluster validity metrics
%Classification Error 

Classification_error_dbscan = match_db/l

% Classification_error_dbscan =
% 
%     0.0291


% 6. Create Clusters from results given by kmeans as per the ground truth


%%first create the clusters

%place the data points in different bins as per kmeans result
kresult1=[]; kresult2=[]; kresult3=[]; kresult4=[]; kresult5=[]; kresult6=[];

for i=1:length(idx)
   j= idx(i,1);
   if j==1
    kresult1= [kresult1, i];
   elseif j==2
           kresult2= [kresult2, i];
   elseif j==3
           kresult3= [kresult3, i] ;
   elseif j==4
          kresult4= [kresult4, i] ;
   elseif j==5
          kresult5= [kresult5, i] ;
   elseif j==6
          kresult6 = [kresult6, i];
    end
    end
              

% now we can see clearly that results of cluster 3, 6 and 2 have no match
% or very few matches
% elements 

% find length of each cluster and see the percentage match
% accordingly we can decide the clusters

clusterNum=zeros(length(kresult6),1);

for i = 1: length(kresult6)
   clusterNum(i,1)= groundbins(kresult6(1,i));

end

sum(clusterNum(:,1)==4)
% 9
sum(clusterNum(:,1)==5)
% 15

sum(clusterNum(:,1)==6)
%6

sum(clusterNum(:,1)==3)
% 8

sum(clusterNum(:,1)==1)
% 8

% We can clearly see that most of the data points in kresult6 belong to groundtruth
% bin5, 1 , 6 and bin 3. Maximum are from bin 5 
% so we will out 1, 2 and 3 to cluster 1 with carb level 0
%and 4,5 and 6 in cluster 5 with carb level 60-80
cluster_1=[]; % carblevel=40-60
for i=1 : length(clusterNum)
    if clusterNum(i,1)==3
        cluster_1=[cluster_1, kresult6(1,i)];
    end
end

%now move the values with bin 1
for i=1 : length(clusterNum)
    if clusterNum(i,1)==1
        cluster_1=[cluster_1, kresult6(1,i)];
    end
end

for i=1 : length(clusterNum)
    if clusterNum(i,1)==2
        cluster_1=[cluster_1, kresult6(1,i)];
    end
end

%now move these points from 4,5 and 6 to cluster 5 

cluster_5=[]; % crab level 60-80
% values of bin4
for i=1 : length(clusterNum)
    if clusterNum(i,1)==4
        cluster_5=[cluster_5, kresult6(1,i)];
    end
end

% values of bin5
for i=1 : length(clusterNum)
    if clusterNum(i,1)==5
        cluster_5=[cluster_5, kresult6(1,i)];
    end
end

%values of bin6 to cluster 5
for i=1 : length(clusterNum)
    if clusterNum(i,1)==6
        cluster_5=[cluster_5, kresult6(1,i)];
    end
end

% now check kresult5

clusterNum=zeros(length(kresult5),1);

for i = 1: length(kresult5)
   clusterNum(i,1)= groundbins(kresult5(1,i));

end

%ignore the ones with zeros as they correspond to meal amount >100 but we
%have to consider only meal amount upto 100

%again merge 5,6 to cluster_5 and points from bin 1,2 and 3 to cluster 1

for i=1 : length(clusterNum)
    if clusterNum(i,1)==6
        cluster_5=[cluster_5, kresult5(1,i)];
    end
end

% values from bin 5
for i=1 : length(clusterNum)
    if clusterNum(i,1)==5
        cluster_5=[cluster_5, kresult5(1,i)];
    end
end


% now values form bin 1, 2 and 3 to cluster_1
for i=1 : length(clusterNum)
    if clusterNum(i,1)==1
        cluster_1=[cluster_1, kresult5(1,i)];
    end
end

for i=1 : length(clusterNum)
    if clusterNum(i,1)==2
        cluster_1=[cluster_1, kresult5(1,i)];
    end
end

for i=1 : length(clusterNum)
    if clusterNum(i,1)==3
        cluster_1=[cluster_1, kresult5(1,i)];
    end
end


% now check kresult4

clusterNum=zeros(length(kresult4),1);

for i = 1: length(kresult4)
   clusterNum(i,1)= groundbins(kresult4(1,i));

end

%again merge 1,2 and 3 to cluster_ and 4&5 to cluster_5
for i=1 : length(clusterNum)
    if clusterNum(i,1)==1
        cluster_1=[cluster_1, kresult4(1,i)];
    end
end

for i=1 : length(clusterNum)
    if clusterNum(i,1)==2
        cluster_1=[cluster_1, kresult4(1,i)];
    end
end


for i=1 : length(clusterNum)
    if clusterNum(i,1)==3
        cluster_1=[cluster_1, kresult4(1,i)];
    end
end

% 4 and 5 to cluster 5
for i=1 : length(clusterNum)
    if clusterNum(i,1)==4
        cluster_5=[cluster_5, kresult4(1,i)];
    end
end

for i=1 : length(clusterNum)
    if clusterNum(i,1)==5
        cluster_5=[cluster_5, kresult4(1,i)];
    end
end

%now considering kresult3

clusterNum=zeros(length(kresult3),1);

for i = 1: length(kresult3)
   clusterNum(i,1)= groundbins(kresult3(1,i));

end
%again ignore the 0. There is only and 1 and 5 so merge 1 to cluster_1 and  5 to cluster_5

for i=1 : length(clusterNum)
    if clusterNum(i,1)==1
        cluster_1=[cluster_1, kresult3(1,i)];
    end
end


% bin 5 to cluster 5

for i=1 : length(clusterNum)
    if clusterNum(i,1)==5
        cluster_5=[cluster_5, kresult3(1,i)];
    end
end


%now considering kresult2

clusterNum=zeros(length(kresult2),1);

for i = 1: length(kresult2)
   clusterNum(i,1)= groundbins(kresult2(1,i));

end

%again we merge 1,2 and 3 to cluster_1 and 4, 5 and 6 to cluster_5

for i=1 : length(clusterNum)
    if clusterNum(i,1)==3
        cluster_1=[cluster_1, kresult2(1,i)];
    end
end

for i=1 : length(clusterNum)
    if clusterNum(i,1)==2
        cluster_1=[cluster_1, kresult2(1,i)];
    end
end
for i=1 : length(clusterNum)
    if clusterNum(i,1)==1
        cluster_1=[cluster_1, kresult2(1,i)];
    end
end

% 4 ,5 and 6 to cluster 5
for i=1 : length(clusterNum)
    if clusterNum(i,1)==4
        cluster_5=[cluster_5, kresult2(1,i)];
    end
end

for i=1 : length(clusterNum)
    if clusterNum(i,1)==5
        cluster_5=[cluster_5, kresult2(1,i)];
    end
end

for i=1 : length(clusterNum)
    if clusterNum(i,1)==6
        cluster_5=[cluster_5, kresult2(1,i)];
    end
end

%now consider kresult1

clusterNum=zeros(length(kresult1),1);

for i = 1: length(kresult1)
   clusterNum(i,1)= groundbins(kresult1(1,i));

end

% ignore 0's and put 4, 5 and 6 in cluster 5 and 1, 2 and 3 in cluster_1


for i=1 : length(clusterNum)
    if clusterNum(i,1)==1
        cluster_1=[cluster_1, kresult1(1,i)];
    end
end

for i=1 : length(clusterNum)
    if clusterNum(i,1)==2
        cluster_1=[cluster_1, kresult1(1,i)];
    end
end

for i=1 : length(clusterNum)
    if clusterNum(i,1)==3
        cluster_1=[cluster_1, kresult1(1,i)];
    end
end

% 4 ,5 and 6 to cluster 5
for i=1 : length(clusterNum)
    if clusterNum(i,1)==4
        cluster_5=[cluster_5, kresult1(1,i)];
    end
end

for i=1 : length(clusterNum)
    if clusterNum(i,1)==5
        cluster_5=[cluster_5, kresult1(1,i)];
    end
end

for i=1 : length(clusterNum)
    if clusterNum(i,1)==6
        cluster_5=[cluster_5, kresult1(1,i)];
    end
end


%Now we have two clusters, cluster 1 and 5. We can analyse both the clusters. 
%The cluster 5 is large so can further divide the clusters.

clusterNum=zeros(length(cluster_1),1);

for i = 1: length(cluster_1)
   clusterNum(i,1)= groundbins(cluster_1(1,i));

end

sum(clusterNum(:,1)==1)

% ans = 52

sum(clusterNum(:,1)==2)

% ans=10

sum(clusterNum(:,1)==3)
% ans = 25

%now analyse cluster_5
clusterNum1=zeros(length(cluster_5),1);

for i = 1: length(cluster_5)
   clusterNum1(i,1)= groundbins(cluster_5(1,i));

end

sum(clusterNum1(:,1)==4)
% ans =
% 
%     25

sum(clusterNum1(:,1)==5)
% ans =
% 
%     61

sum(clusterNum1(:,1)==6)
% ans =
% 
%     21
    

%We can create one new cluster and distribute the values as follows
% since %points with 2 are very few
% We can put 1 and 2 in cluster1 with carblevels 0

clusterOne=[]; % crab level 0
for i=1 :length(clusterNum)
    if clusterNum(i,1)==1
        clusterOne=[clusterOne, cluster_1(1,i)];
    end
end

for i=1 :length(clusterNum)
    if clusterNum(i,1)==2
        clusterOne=[clusterOne, cluster_1(1,i)];
    end
end

% we can place the 3 and 4 in cluster2 40- 60

clusterTwo=[]; % crab level 40-60
for i=1 :length(clusterNum)
    if clusterNum(i,1)==3
        clusterTwo=[clusterTwo, cluster_1(1,i)];
    end
end

%adding points 4 from cluster_5 to second cluster clusterTwo
for i=1 :length(clusterNum1)
    if clusterNum1(i,1)==4
        clusterTwo = [clusterTwo, cluster_5(1,i)];
    end
end

% and points in 5 and 6 in cluster3 60-80

clusterThree=[]; % crab level 60-80
for i=1:length(clusterNum1)
    if clusterNum1(i,1)==5
        clusterThree = [clusterThree, cluster_5(1,i)];
    end
end


for i=1:length(clusterNum1)
    if clusterNum1(i,1)==6
        clusterThree = [clusterThree, cluster_5(1,i)];
    end
end

length(clusterOne)
% ans =
% 
%     62

length(clusterTwo)
% ans =
% 
%     50

length(clusterThree)
% ans =
% 
%     82


%our three clusters seems to be pretty balanced

% now we need to find the centroid of all the three clusters and choose an
% optimal value of k such that we can find the nearest cluster for any new
% test data

% we can consider the centroids of bin 1, 3 and 5 as the new clusters are
% formed by merging the points into them

centroid_clusters= [c(1,:);c(2,:);c(3,:)];
% write the centroids of clusters from ground truth to a csv file
writematrix(centroid_clusters, 'centroid_clusters.csv');

%write the centroids of clusters from kmeans to a csv file
writematrix(c, 'centroid_kmeans.csv');

%find the feature matrix for test data 
testMealData=csvread('mealDataX.csv',0,0);
test_matrix= find_features(testMealData);

% we can use Euclidean distance to find the distance of test data from each
% cluster and return the nearest cluster
[~,idx_test] = pdist2(centroid_clusters,test_matrix,'euclidean','Smallest',1);


% check the nearest cluster given by kmeans for first 20 rows in train data
[~,idx_train] = pdist2(c, feature_matrix(1:20,:),'euclidean','Smallest',1)

%gives the nearest cluster number for each 

% idx_train =
% 
%      2     4     4     4     4     6     6     6     1     4     5     6     1     1     4     4     5     6     6     1
% 

% check the nearest cluster for train data from new three clusters which we
% created as per the ground truth
[~,idx_train_c] = pdist2(centroid_clusters, feature_matrix(1:30,:),'euclidean','Smallest',1)

% idx_train_c =
% 
%   Columns 1 through 20
% 
%      2     1     1     1     1     1     1     1     1     1     1     2     1     1     1     1     1     1     1     1
% 
%   Columns 21 through 30
% 
%      1     1     1     1     3     3     3     1     1     1


