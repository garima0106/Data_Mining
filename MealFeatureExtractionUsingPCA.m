
% time series data - patient glucose level

%read all the csv data files
T1=readtable('CGMSeriesLunchPat1.csv');
T2=readtable('CGMSeriesLunchPat2.csv');
T3=readtable('CGMSeriesLunchPat3.csv');
T4=readtable('CGMSeriesLunchPat4.csv');
T5= readtable('CGMSeriesLunchPat5.csv');

New_CGM4=T4(:,1:31);

% combine the patient data
CGMSeries= [T1;T2;T3;New_CGM4;T5];

S1=readtable('CGMDatenumLunchPat1.csv');
S2=readtable('CGMDatenumLunchPat2.csv');
S3=readtable('CGMDatenumLunchPat3.csv');
S4=readtable('CGMDatenumLunchPat4.csv');
S5=readtable('CGMDatenumLunchPat5.csv');

S4_New= S4(:,1:31);

CGMData= [S1;S2;S3;S4_New;S5];

%col 31 is mostly Nan
CGMSeries= CGMSeries(:,1:30);
CGMData=CGMData(:,1:30);

plot(CGMData(5,:), CGMSeries(5,:));

% convert the tables to array
CGMData=table2array(CGMData);
CGMSeries=table2array(CGMSeries);

% find how many rows have Nan or missing values
 rows_to_be_deleted_data=0;
 rows_to_be_deleted_series=0;
for i=1:length(CGMSeries(:,1))
   series_nan=0;
   data_nan=0;
   for j=1:length(CGMSeries(i,:)) 
        if isnan(CGMSeries(i,j))
          series_nan= series_nan+1;
          
        end
        if isnan(CGMData(i,j))
            data_nan=data_nan+1;
        end
        
       
   end
    if series_nan>= 0.2*length(CGMSeries(i,:))
          rows_to_be_deleted_series = rows_to_be_deleted_series+1;
          
    end
    
    if data_nan>= 0.2*length(CGMData(i,:))
          rows_to_be_deleted_data = rows_to_be_deleted_data+1;
    end
   %CGMSeries(i,j)=[]
end
   
% remove the same rows from CGMseries and CGMdata tables
%add a new coulmn to keep a counter of rows
NewCol=[];
for i=1:216
 NewCol=[NewCol;i]
end
CGMSeries(:,31)= NewCol;

% remove missing values
CGMSeries_New= rmmissing(CGMSeries);

%remove same rows from CGMData table
for i= CGMSeries_New(:,31)
CGMData_New=CGMData(i,:)
end

CGMSeries_New=CGMSeries_New(:,1:30);

len=length(CGMSeries_New(:,:));

% finding features
feature_matrix= zeros(len,50);

% first feature - Statistical features

feature_first_stat = zeros(len,26);

% sliding mean window - every 20 min - 4 windows
k=1;
start_window=1;
end_window=4;
while k<=14
for i=1:size(CGMSeries_New,1)
    feature_first_stat(i,k)= mean(CGMSeries_New(i,start_window:end_window)); 
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
for i=1:size(CGMSeries_New,1)
    feature_first_stat(i,k)= std(CGMSeries_New(i,start_window:end_window)); 
end
start_window=start_window+3;
end_window=end_window+3;
k=k+1;
end

bar(1:1:9, mean(feature_first_stat(:,15:23),1))

% 3. Overall Mean and Variance
% 4. Skewness
feature_first_stat(:,24)=mean(CGMSeries_New,2); %mean
feature_first_stat(:,25)=std(CGMSeries_New,0, 2); %standard deviation
feature_first_stat(:,26)= skewness(CGMSeries_New,0, 2); %skewness

feature_first_stat=normalize(feature_first_stat);

% second feature - polynomial fit
feature_second_poly = zeros(len,6);
for i = 1:size(CGMSeries_New,1)
    feature_second_poly(i,1:6)=polyfit((1:1:30),flip(CGMSeries_New(i,:)),5);
end
%normalize the data
feature_second_poly=normalize(feature_second_poly);

bar(1:1:6, mean(feature_second_poly(:,1:6),1));

% third feature - fourier transform
feature_third_fft= zeros(len,5);
df1=fft(CGMSeries_New(:,:));

% magnitude of fft values
mag_df1= abs(df1);
% this creates 186 bins
plot(mag_df1);


% drop the first bin
mag_df1= mag_df1(2:186,1:30);

% we are interested in only bins with large magnitude
for i = 1:size(CGMSeries_New,1)-1
     feature_third_fft(i,1:8)= max(mag_df1(i,1:8));
end

feature_third_fft=normalize(feature_third_fft);

%fourth feature
feature_fourth_velocity= zeros(len,5);

for i = 1:size(CGMSeries_New,1)
        CGMVelocity(i,1:29) = CGMSeries_New(i,1:end-1)-CGMSeries_New(i,2:end);
    
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
                x1= CGMVelocity(i,j);
                x2= CGMVelocity(i,j+1); 
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
feature_matrix=feature_matrix(1:185,:);
feature_matrix(:,1:26)= feature_first_stat(1:185,:);
feature_matrix(:,27:32)=feature_second_poly(1:185,:);
feature_matrix(:,33:40)=feature_third_fft(1:185,:);
feature_matrix(:,41:44)=feature_fourth_velocity(1:185,1:4);

% approximate entropy - to recognize patterns
%find peaks - to find peak values in the data and take a mean of those
% find min, max
% find mean of difference of two values
% find range = max-min
%some_more_features_1=zeros(len-1,6);
for i = 1:size(CGMSeries_New-1,1)
    some_more_features_1(i,1)= approximateEntropy(CGMSeries_New(i,:)); %approx entropy
    %Peak Value Analysis
    some_more_features_2(i,1)= size(findpeaks(CGMSeries_New(i,:), 'MinPeakProminence',1,'Annotate','extents'),2);
    some_more_features_3(i,1)= min(CGMSeries_New(i,:)); %min value
    some_more_features_4(i,1)= max(CGMSeries_New(i,:)); %max value
    some_more_features_5(i,1)= mean(abs(diff(CGMSeries_New(i,:)))); %mean of difference between 2 values
    some_more_features_6(i,1)=max(CGMSeries_New(i,:))-min(CGMSeries_New(i,:)); % range
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

feature_matrix=feature_matrix(1:185,:);

%use pca to extract important features in direction of maximum variance

%[coeff, score,latent]=pca(feature_matrix(:,1:50));


feature_extract_new=pca(feature_matrix);

% pick the top 5 eigen vectors to get the final feature matrix
%construct new feature matrix

%feature_e= feature_matrix*feature_extract_new;

% new feature matrix
feature_final= feature_matrix*feature_extract_new(:,1:5);


plot(CGMData_New(5,:), CGMSeries_New(5,:));
plot(1:1:185,feature_final(:,1)); 
plot(1:1:185,feature_final(:,2));

plot(1:1:185,feature_final(:,3));
plot(1:1:185,feature_final(:,4));
plot(1:1:185,feature_final(:,5));


    %peak analysis
load sunspot.dat
%X=findpeaks(CGMSeries_New)

%{
    
    peak value analysis
for i = 1:size(CGMSeries_New,1)
     Y(i,1:size(findpeaks(CGMSeries_New(i,:))))=findpeaks(CGMSeries_New(i,:));
   
end
}%
%Y= findpeaks(CGMSeries_New, 'minpeakheight', 250);

%{
%find fourier transform to find the frequrncy component 
Fs = 10000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector

freq=Fs*(0:(L/2))/L;

df1=fft(CGMSeries_New(:,:));

% magnitude of fft values
mag_df1= abs(df1);
% this creates 186 bins

plot(mag_df1);

% we are interested in only first 8 bins
mag_df2= mag_df1(:,1:8);

% find the maximum frequency
for i = 1:size(CGMSeries_New,1)
     fIndex=max(mag_df2(i,:));
     
end


for i = 1:size(CGMSeries_New,1)
     feature_matrix(i,12:14)= fftmeal(i,1:3);
end

fftmeal_after_meal_consumption= abs(fft(CGMSeries_New(:,15:23)));

for i = 1:size(CGMSeries_New,1)
     feature_matrix(i,12:19)= mag_df2(i,:);
end
%}




        
        