
% time series data - patient glucose level

%read all the csv data files
NM1=csvread('Nomeal1.csv',0,0);
NM2=csvread('Nomeal2.csv',0,0);
NM3=csvread('Nomeal3.csv',0,0);
NM4=csvread('Nomeal4.csv',0,0);
NM5= csvread('Nomeal5.csv',0,0);


% combine the patient data
NoMealData= [NM1;NM2;NM3;NM4;NM5];

M1=csvread('mealData1.csv',0,0);
M2=csvread('mealData2.csv',0,0);
M3=csvread('mealData3.csv',0,0);
M4=csvread('mealData4.csv',0,0);
M5=csvread('mealData5.csv',0,0);



MealData= [M1;M2;M3;M4;M5];

%col 31 is mostly zero in meal data
MealData= MealData(:,1:30);

%remove the missing or NAN data
MealData= rmmissing(MealData);
NoMealData=rmmissing(NoMealData);


row_to_be_deleted=0;
for i=1:length(MealData(:,1))
    zero_data=0;
    for j=1:length(MealData(i,:))
        if MealData(i,j)==0
            zero_data=zero_data+1;
        end
    end
    if zero_data>=1
       row_to_be_deleted=row_to_be_deleted+1;
      
    end  
end

%set all zeros to NaN
MealData(MealData==0) =NaN;

%remove NaN from the data
MealData= rmmissing(MealData);

rows_to_be_deleted=0;
for i=1:length(NoMealData(:,1))
    zero_data=0;
    for j=1:length(NoMealData(i,:))
        if NoMealData(i,j)==0
            zero_data=zero_data+1;
        end
    end
    if zero_data>=1
       rows_to_be_deleted=rows_to_be_deleted+1;
      
    end  
end

%set all zeros to NaN
NoMealData(NoMealData==0) =NaN;

%remove NaN from the data
NoMealData= rmmissing(NoMealData);


%combine the data
%add on;e class label for Meal and NoMeal Data



len=length(MealData);
% finding features
feature_matrix= zeros(len,50);

% first feature - Statistical features

feature_first_stat = zeros(len,26);

% sliding mean window - every 20 min - 4 windows
k=1;
start_window=1;
end_window=4;
while k<=14
for i=1:size(MealData,1)
    feature_first_stat(i,k)= mean(MealData(i,start_window:end_window)); 
end
start_window=start_window+2;
end_window=end_window+2;
k=k+1;
end

bar(1:1:14, mean(feature_first_stat(:,1:14),1))
%sliding variance window- every 20 min - 4 windows
k=15;
start_window=1;
end_window=4;
while k<=23
for i=1:size(MealData,1)
    feature_first_stat(i,k)= std(MealData(i,start_window:end_window)); 
end
start_window=start_window+3;
end_window=end_window+3;
k=k+1;
end

bar(1:1:9, mean(feature_first_stat(:,15:23),1))

% 3. Overall Mean and Variance
% 4. Skewness
feature_first_stat(:,24)=mean(MealData,2); %mean
feature_first_stat(:,25)=std(MealData,0, 2); %standard deviation
feature_first_stat(:,26)= skewness(MealData,0, 2); %skewness

feature_first_stat=normalize(feature_first_stat);

% second feature - polynomial fit
feature_second_poly = zeros(len,6);
for i = 1:size(MealData,1)
    feature_second_poly(i,1:6)=polyfit((1:1:30),flip(MealData(i,:)),5);
end
%normalize the data
feature_second_poly=normalize(feature_second_poly);

bar(1:1:6, mean(feature_second_poly(:,1:6),1));

% third feature - fourier transform
feature_third_fft= zeros(len,5);
df1=fft(MealData(:,:));

% magnitude of fft values
mag_df1= abs(df1);
% this creates 216 bins
plot(mag_df1);


% drop the first bin
mag_df1= mag_df1(2:216,1:30);

% we are interested in only bins with large magnitude
for i = 1:size(MealData,1)-1
     feature_third_fft(i,1:8)= max(mag_df1(i,1:8));
end

feature_third_fft=normalize(feature_third_fft);

%fourth feature
feature_fourth_velocity= zeros(len,5);

for i = 1:size(MealData,1)
        MealVelocity(i,1:29) = MealData(i,1:end-1)-MealData(i,2:end);
    
end

% find the zero crossings
 zero_crossing_count=zeros(len,5);
    for i = 1:len
        start_window=1;
        bin_window =5;
        bin=1;
        while(bin<=5 && bin_window<=28 )
            count=0;
            for j= start_window:bin_window
                x1= MealVelocity(i,j);
                x2= MealVelocity(i,j+1); 
                if (x1<0 && x2>0)
                    count = count+ 1;
                else
                   if (x1>0 && x2<0)
                    count = count+ 1;
                   end
                end  
            end
            zero_crossing_count(i,bin)=count;
            start_window=bin_window+1;
            bin_window =start_window+5;
            bin=bin+1;
           
        end
    end
    
   feature_fourth_velocity = zero_crossing_count(:,1:4);
  %normalize
  feature_fourth_velocity= normalize(feature_fourth_velocity);
  
  bar(1:1:4, sum(zero_crossing_count(:,1:4),1));
% Concatenate all the features to feature matrix
feature_matrix=feature_matrix(1:216,:);
feature_matrix(:,1:26)= feature_first_stat(1:216,:);
feature_matrix(:,27:32)=feature_second_poly(1:216,:);
feature_matrix(:,33:40)=feature_third_fft(1:216,:);
feature_matrix(:,41:44)=feature_fourth_velocity(1:216,1:4);

% approximate entropy - to recognize patterns
%find peaks - to find peak values in the data and take a mean of those
% find min, max
% find mean of difference of two values
% find range = max-min
%some_more_features_1=zeros(len-1,6);
for i = 1:size(MealData-1,1)
    some_more_features_1(i,1)= approximateEntropy(MealData(i,:)); %approx entropy
    %Peak Value Analysis
    some_more_features_2(i,1)= size(findpeaks(MealData(i,:), 'MinPeakProminence',1,'Annotate','extents'),2);
    some_more_features_3(i,1)= min(MealData(i,:)); %min value
    some_more_features_4(i,1)= max(MealData(i,:)); %max value
    some_more_features_5(i,1)= mean(abs(diff(MealData(i,:)))); %mean of difference between 2 values
    some_more_features_6(i,1)=max(MealData(i,:))-min(MealData(i,:)); % range
end

%normalize
some_more_features_1=normalize(some_more_features_1);
some_more_features_2=normalize(some_more_features_2);
some_more_features_3=normalize(some_more_features_3);
some_more_features_4=normalize(some_more_features_4);
some_more_features_5=normalize(some_more_features_5);
some_more_features_6=normalize(some_more_features_6);

feature_matrix(:,45)=some_more_features_1;
feature_matrix(:,46)=some_more_features_2;
feature_matrix(:,47)=some_more_features_3;
feature_matrix(:,48)=some_more_features_4;
feature_matrix(:,49)=some_more_features_5;
feature_matrix(:,50)=some_more_features_6;

feature_matrix=feature_matrix(1:216,:);

%use pca to extract important features in direction of maximum variance

%[coeff, score,latent]=pca(feature_matrix(:,1:50));


feature_extract_new=pca(feature_matrix);

% pick the top 5 eigen vectors to get the final feature matrix
%construct new feature matrix

%feature_e= feature_matrix*feature_extract_new;

% new feature matrix
feature_final= feature_matrix*feature_extract_new(:,1:5);



plot(1:1:216,feature_final(:,1)); 
plot(1:1:216,feature_final(:,2));

plot(1:1:216,feature_final(:,3));
plot(1:1:216,feature_final(:,4));
plot(1:1:216,feature_final(:,5));

% Features for NoMeal Data
len1=length(NoMealData);
 feature_matrix_NoMeal= zeros(len1,50);
% first feature - Statistical features

feature_first_stat_1 = zeros(len1,26);

% sliding mean window - every 20 min - 4 windows
k=1;
start_window=1;
end_window=4;
while k<=14
for i=1:size(NoMealData,1)
    feature_first_stat_1(i,k)= mean(NoMealData(i,start_window:end_window)); 
end
start_window=start_window+2;
end_window=end_window+2;
k=k+1;
end

bar(1:1:14, mean(feature_first_stat_1(:,1:14),1))
%sliding variance window- every 20 min - 4 windows
k=15;
start_window=1;
end_window=4;
while k<=23
for i=1:size(NoMealData,1)
    feature_first_stat_1(i,k)= std(NoMealData(i,start_window:end_window)); 
end
start_window=start_window+3;
end_window=end_window+3;
k=k+1;
end

bar(1:1:9, mean(feature_first_stat_1(:,15:23),1))

% 3. Overall Mean and Variance
% 4. Skewness
feature_first_stat_1(:,24)=mean(NoMealData,2); %mean
feature_first_stat_1(:,25)=std(NoMealData,0, 2); %standard deviation
feature_first_stat_1(:,26)= skewness(NoMealData,0, 2); %skewness

feature_first_stat_1=normalize(feature_first_stat_1);

% second feature - polynomial fit
feature_second_poly_1 = zeros(len1,6);
for i = 1:size(NoMealData,1)
    feature_second_poly_1(i,1:6)=polyfit((1:1:30),flip(NoMealData(i,:)),5);
end
%normalize the data
feature_second_poly_1=normalize(feature_second_poly_1);

bar(1:1:6, mean(feature_second_poly_1(:,1:6),1));

% third feature - fourier transform
feature_third_fft_1= zeros(len1,5);
df_N=fft(NoMealData(:,:));

% magnitude of fft values
mag_dfN= abs(df_N);
% this creates 216 bins
plot(mag_dfN);


% drop the first bin
mag_dfN= mag_dfN(2:219,1:30);

% we are interested in only bins with large magnitude
for i = 1:size(NoMealData,1)-1
     feature_third_fft_1(i,1:8)= max(mag_dfN(i,1:8));
end

feature_third_fft_1=normalize(feature_third_fft_1);

%fourth feature
feature_fourth_velocity_1= zeros(len,5);

for i = 1:size(NoMealData,1)
        NoMealVelocity(i,1:29) = NoMealData(i,1:end-1)-NoMealData(i,2:end);
    
end

% find the zero crossings
 zero_crossing_count_1=zeros(len1,5);
    for i = 1:len
        start_window=1;
        bin_window =5;
        bin=1;
        while(bin<=5 && bin_window<=28 )
            count=0;
            for j= start_window:bin_window
                x1= NoMealVelocity(i,j);
                x2= NoMealVelocity(i,j+1); 
                if (x1<0 && x2>0)
                    count = count+ 1;
                else
                   if (x1>0 && x2<0)
                    count = count+ 1;
                   end
                end  
            end
            zero_crossing_count_1(i,bin)=count;
            start_window=bin_window+1;
            bin_window =start_window+5;
            bin=bin+1;
           
        end
    end
    
   feature_fourth_velocity_1 = zero_crossing_count_1(:,1:4);
  %normalize
  feature_fourth_velocity_1= normalize(feature_fourth_velocity_1);
  
  bar(1:1:4, sum(zero_crossing_count_1(:,1:4),1));
% Concatenate all the features to feature matrix
feature_matrix_NoMeal=feature_matrix_NoMeal(1:219,:);
feature_matrix_NoMeal(:,1:26)= feature_first_stat_1(1:219,:);
feature_matrix_NoMeal(:,27:32)=feature_second_poly_1(1:219,:);
feature_matrix_NoMeal(:,33:40)=feature_third_fft_1(1:219,:);
feature_matrix_NoMeal(:,41:44)=feature_fourth_velocity_1(1:219,1:4);

% approximate entropy - to recognize patterns
%find peaks - to find peak values in the data and take a mean of those
% find min, max
% find mean of difference of two values
% find range = max-min
%some_more_features_1=zeros(len-1,6);
for i = 1:size(NoMealData-1,1)
    some_more_featuresN_1(i,1)= approximateEntropy(NoMealData(i,:)); %approx entropy
    %Peak Value Analysis
    some_more_featuresN_2(i,1)= size(findpeaks(NoMealData(i,:), 'MinPeakProminence',1,'Annotate','extents'),2);
    some_more_featuresN_3(i,1)= min(NoMealData(i,:)); %min value
    some_more_featuresN_4(i,1)= max(NoMealData(i,:)); %max value
    some_more_featuresN_5(i,1)= mean(abs(diff(NoMealData(i,:)))); %mean of difference between 2 values
    some_more_featuresN_6(i,1)=max(NoMealData(i,:))-min(NoMealData(i,:)); % range
end

%normalize
some_more_featuresN_1=normalize(some_more_featuresN_1);
some_more_featuresN_2=normalize(some_more_featuresN_2);
some_more_featuresN_3=normalize(some_more_featuresN_3);
some_more_featuresN_4=normalize(some_more_featuresN_4);
some_more_featuresN_5=normalize(some_more_featuresN_5);
some_more_featuresN_6=normalize(some_more_featuresN_6);

feature_matrix_NoMeal(:,45)=some_more_featuresN_1;
feature_matrix_NoMeal(:,46)=some_more_featuresN_2;
feature_matrix_NoMeal(:,47)=some_more_featuresN_3;
feature_matrix_NoMeal(:,48)=some_more_featuresN_4;
feature_matrix_NoMeal(:,49)=some_more_featuresN_5;
feature_matrix_NoMeal(:,50)=some_more_featuresN_6;

feature_matrix_NoMeal=feature_matrix_NoMeal(1:219,:);

%use pca to extract important features in direction of maximum variance

%[coeff, score,latent]=pca(feature_matrix(:,1:50));


feature_extract_new_NoMeal=pca(feature_matrix_NoMeal);

% pick the top 5 eigen vectors to get the final feature matrix
%construct new feature matrix

%feature_e= feature_matrix*feature_extract_new;

% new feature matrix
feature_final_NoMeal= feature_matrix_NoMeal*feature_extract_new_NoMeal(:,1:5);


 
%add the label to meal features and Nomeal features 
%after pca and combine the data for training
label(1:216,1)=1;
Meal_features= [feature_final label];

label1(1:219,1)= -1;

NoMeal_features= [feature_final_NoMeal label1];

Trainingdata= [Meal_features;NoMeal_features];

%randomize the data
random_training = Trainingdata(randperm(size(Trainingdata, 1)), :);

% train the classifier using SVM

SVMmodel= fitcsvm(random_training(:,1:5), random_training(:,6), 'KernelFunction', 'rbf');


%training classification error
L = loss(SVMmodel,ran_train_to_test(:,1:5),ran_train_to_test(:,6));

%use cross validation
CVSVMmodel= crossval(SVMmodel,'KFold',5);

%predict after cross fold

[valPred, valScores]=kfoldPredict(CVSVMmodel);
%find cross validation loss
val_error=kfoldLoss(CVSVMmodel, 'LossFun', 'ClassifError');

%find model accuracy
val_accuracy=1-val_error

%function to test the model and find accuracy
%ran_train= rand(435,6);
k=randperm(435);
ran_train_to_test=random_training(k(1:250),:);
acc=0;
l=length(ran_train_to_test);
res=predict(SVMmodel,ran_train_to_test(:,1:5));

for i=1:length(res)
   if res(i)==ran_train_to_test(i,6)
      acc=acc+1;  
   end
end 
accuracy=(acc/l)*100

%save the model
saveCompactModel(SVMmodel,'my_model');




