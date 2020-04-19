
% test file to test the model output

file = subdir('C:\Users\garima\Documents\Matlab\TestData.csv');

Testfilename=csvread('TestData.csv',0,0);

%load the saved model 
SModel=load('my_model')

%extract the features of test file
% function 'feature_extraction' is given below in the same file
feature_test=feature_extraction(Testfilename);

%here meal label is 1 and nomeal label is -1

%function to test the model 

SModel=loadLearnerForCoder('my_model')
predictionResults=predict(SModel,feature_test(:,1:5));


%add the predicted labels to the feature_test
feature_test_new= [feature_test predictionResults ];


%define the actual values for label
%actual_label 
%function to find accuracy
acc=0;
l=length(feature_test);
for i=1:length(res)
   if feature_test(i,6)==actual_label(i)
      acc=acc+1;  
   end
end 
predictionAccuracy=(acc/l)*100;

%confusion matrix

%confusion matrix
[confMat,order] = confusionmat(actual_label(:,1),feature_test(:,6));

% recall and precision
for i =1: size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(i,:));
    precision(i)=confMat(i,i)/sum(confMat(:,i));
    
end

Recall = sum(recall)/size(confMat,1);
Precision=sum(precision)/size(confMat,1);

Recall = sum(recall)/size(confMat,1);
Precision=sum(precision)/size(confMat,1);

%F-score
F_score=2*Recall*Precision/(Precision+Recall); 


% function to extract the features

function feature_test = feature_extraction (Testfilename)
         Testfilename= rmmissing(Testfilename);
         Testfilename(Testfilename==0) =NaN;

        %remove NaN from the data
         Testfilename= rmmissing(Testfilename);
         
         len=length(Testfilename);
            % finding features
        feature_matrix= zeros(len,50);
        feature_first_stat = zeros(len,26);

        % sliding mean window - every 20 min - 4 windows
        k=1;
        start_window=1;
        end_window=4;
        while k<=14
        for i=1:size(Testfilename,1)
            feature_first_stat(i,k)= mean(Testfilename(i,start_window:end_window)); 
        end
        start_window=start_window+2;
        end_window=end_window+2;
        k=k+1;
        end

        %sliding variance window- every 20 min - 4 windows
        k=15;
        start_window=1;
        end_window=4;
        while k<=23
        for i=1:size(Testfilename,1)
            feature_first_stat(i,k)= std(Testfilename(i,start_window:end_window)); 
        end
        start_window=start_window+3;
        end_window=end_window+3;
        k=k+1;
        end
        
        feature_first_stat(:,24)=mean(Testfilename,2); %mean
        feature_first_stat(:,25)=std(Testfilename,0, 2); %standard deviation
        feature_first_stat(:,26)= skewness(Testfilename,0, 2); %skewness

        feature_first_stat=normalize(feature_first_stat);

        % second feature - polynomial fit
        feature_second_poly = zeros(len,6);
        for i = 1:size(Testfilename,1)
            feature_second_poly(i,1:6)=polyfit((1:1:30),flip(Testfilename(i,:)),5);
        end
        %normalize the data
        feature_second_poly=normalize(feature_second_poly);

        % third feature - fourier transform
        feature_third_fft= zeros(len,5);
        df1=fft(Testfilename(:,:));

        % magnitude of fft values
        mag_df1= abs(df1);
        % this creates 216 bins
        plot(mag_df1);

        % drop the first bin
        mag_df1= mag_df1(2:len,1:30);

        % we are interested in only bins with large magnitude
        for i = 1:size(Testfilename,1)-1
             feature_third_fft(i,1:8)= max(mag_df1(i,1:8));
        end

        feature_third_fft=normalize(feature_third_fft);

        %fourth feature
        feature_fourth_velocity= zeros(len,5);

        for i = 1:size(Testfilename,1)
                MealVelocity(i,1:29) = Testfilename(i,1:end-1)-Testfilename(i,2:end);

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
        feature_matrix=feature_matrix(1:len,:);
        feature_matrix(:,1:26)= feature_first_stat(1:len,:);
        feature_matrix(:,27:32)=feature_second_poly(1:len,:);
        feature_matrix(:,33:40)=feature_third_fft(1:len,:);
        feature_matrix(:,41:44)=feature_fourth_velocity(1:len,1:4);

        % approximate entropy - to recognize patterns
        %find peaks - to find peak values in the data and take a mean of those
        % find min, max
        % find mean of difference of two values
        % find range = max-min
        %some_more_features_1=zeros(len-1,6);
        for i = 1:size(Testfilename-1,1)
            some_more_features_1(i,1)= approximateEntropy(Testfilename(i,:)); %approx entropy
            %Peak Value Analysis
            some_more_features_2(i,1)= size(findpeaks(Testfilename(i,:), 'MinPeakProminence',1,'Annotate','extents'),2);
            some_more_features_3(i,1)= min(Testfilename(i,:)); %min value
            some_more_features_4(i,1)= max(Testfilename(i,:)); %max value
            some_more_features_5(i,1)= mean(abs(diff(Testfilename(i,:)))); %mean of difference between 2 values
            some_more_features_6(i,1)=max(Testfilename(i,:))-min(Testfilename(i,:)); % range
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

        feature_matrix=feature_matrix(1:len,:);

        %use pca to extract important features in direction of maximum variance

      


        feature_extract_new=pca(feature_matrix);

        % pick the top 5 eigen vectors to get the final feature matrix
        %construct new feature matrix

        %feature_e= feature_matrix*feature_extract_new;

        % new feature matrix
        feature_test= feature_matrix*feature_extract_new(:,1:5);

end
