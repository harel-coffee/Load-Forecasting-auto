% clear all
clear;  
% clc

%% load data
% path='R:\Haris\SVD load forecasting\datasets\hour and customers agg load\Swe\';
path='R:\Haris\SVD load forecasting\datasets\hour and customers agg load\single_file\Swe\';
% listing = dir(path);
fileList = getAllFiles(path);

% path2 = 'R:\Pasha\Matlab Code Forecasting\Dataset\Weather\swe\';
path2 = 'R:\Pasha\Matlab Code Forecasting\Dataset\Weather\single_file\swe\';
File_Name = getAllFiles(path2);

min_mape = 200;
min_mae = 1000;
min_rmse = 1000;
for itter = 1:1
itter
    %Sweden
    err=cell(50,1);
    temp_count_1 = 1;
    mape_matrix = zeros(length(File_Name(:,1))+1,length(File_Name(:,1))+1);
    smape_matrix = zeros(length(File_Name(:,1))+1,length(File_Name(:,1))+1);
    rmse_matrix = zeros(length(File_Name(:,1))+1,length(File_Name(:,1))+1);

    mape_matrix(1,1) = 0;
    smape_matrix(1,1) = 0;
    rmse_matrix(1,1) = 0;

    row_temp=1;
    col_temp=1;
    row_temp_2 = 1;
    %%
    for z=1:length(fileList)  
        z;
    Original_Dataset_initial = csvread(string(fileList(z,1)));
    dataset_name_string = 'Sweden';
     [rr,cc]=size(Original_Dataset_initial);       

      if  cc== 582
        dimensions= 300;
      elseif  cc== 16
        dimensions= 10;
     elseif cc== 8
         dimensions=8;
     elseif cc== 4
         dimensions=4;
     elseif cc== 2
         dimensions=2;
      elseif cc== 1
         dimensions=1;       
     end
    clust_iterations=100;

     if  rr== 17544
        end_test_row= 8784;
        total_predicted_rows = 4104;
     elseif rr== 8772
         end_test_row= (8784/2);
         total_predicted_rows = (4104/2);
     elseif rr== 4386
         end_test_row= (8784/4);
         total_predicted_rows = (4104/4);
     elseif rr== 1462
         end_test_row= (8784/12);
         total_predicted_rows = (4104/12);
      elseif rr== 731
         end_test_row= (8784/24);  
         total_predicted_rows = (4104/24);
     end
    % end_test_row = 8783;
    % total_predicted_rows = 4104;
    % Cluster_Range = [80 100 150];
    Cluster_Range = 80;
    % all_powers_clusters_error = zeros(10, length(Cluster_Range));
    % for n=1:10
    %     for clust_loop = 1:length(Cluster_Range)
    %     clearvars -except n clust_loop Cluster_Range all_powers_clusters_error


        Original_Dataset = Original_Dataset_initial;

        % Original_Dataset(:,135)=[];
        % Original_Dataset(:,170)=[];

        POWER = 4;
        Original_Dataset = Original_Dataset.^(1/POWER);

    %     [min(min(Original_Dataset)) median(median(Original_Dataset)) max(max(Original_Dataset)) mean(mean(Original_Dataset)) ];

        tic

            Starting_Row = 1;
            Ending_Row = end_test_row;
            Predicted_Row = Ending_Row+1;

            Data = Original_Dataset(Starting_Row:Ending_Row,:);


            Data_check = csvread(string(File_Name(temp_count_1,1)),1,0);
            max_temperature = max(Data_check(:,5));
            max_wind_speed = max(Data_check(:,7));
            max_humidity = max(Data_check(:,8));

            min_temperature = min(Data_check(:,5));
            min_wind_speed = min(Data_check(:,7));
            min_humidity = min(Data_check(:,8));

            Data_check(:,5) = (Data_check(:,5)-min_temperature)/(max_temperature-min_temperature);
            Data_check(:,7) = (Data_check(:,7)-min_wind_speed)/(max_wind_speed-min_wind_speed);
            Data_check(:,8) = (Data_check(:,8)-min_humidity)/(max_humidity-min_humidity);

            Data1 = Data_check(Starting_Row:Ending_Row,:);


           if z==6
                temp_count_1 = temp_count_1+1;
            elseif z==12
                temp_count_1 = temp_count_1+1;
                elseif z==18
                temp_count_1 = temp_count_1+1;
                elseif z==24
                temp_count_1 = temp_count_1+1;
                elseif z==30
                temp_count_1 = temp_count_1+1;
    %             elseif z==30
    %             temp_count_1 = temp_count_1+1;
              end

            dim_red=dimensions;

            SVD_normalized = zeros(length(Data(:,1)),(length(Data(1,:)))+3);
            SVD_normalized(:,1:length(Data(1,:))) = Data;
            SVD_normalized(:,length(Data(1,:))+1:(length(Data(1,:)))+3) =  [Data1(:,5) Data1(:,7) Data1(:,8)];
            max_normalized_by_column_SVD = bsxfun(@rdivide,SVD_normalized,...
                max(SVD_normalized,[],1));

            [U,S,V] = svd(max_normalized_by_column_SVD,'econ');
            S_reduced=S(:,1:dim_red);
            U_reduced=U;
            V_reduced=V(:,1:dim_red)';

            A_reduced=U_reduced*S_reduced;
            % A1=U_reduced*S_reduced*V_reduced;
            SVD_File_1 = A_reduced;

        %     SVD_File = SVD_File_1(Starting_Row_1(p):Ending_Row,:);
            SVD_File = SVD_File_1;

                Cluster_Column = kmeans(SVD_File,Cluster_Range,'MaxIter',2000,'Replicates',clust_iterations,'EmptyAction','drop');

                %% one hot encoding
                % Clustered rows are stored in the excel file, we read cluster number from file
                cluster_number=max(Cluster_Column);

                % cluster_encoding Matrix contains the name of each cluster. Then we
                % compare it with the one hot encoding vector of the time which we are
                % predicting
                non_empty_clusters=0;
                for j=1:cluster_number
                cluster_row_index= find(Cluster_Column==j);
                    cluster_length= length(cluster_row_index);
                    if(cluster_length~=0)
                        non_empty_clusters =  non_empty_clusters+1;
                    end
                end
                cluster_encoding=zeros(non_empty_clusters,79);

                crnt_cluster_ID=1;
                cluster_Sizes = zeros(non_empty_clusters,1);
                for j=1:cluster_number

                    hour_of_the_day=zeros(1,24);
                    day_of_the_week=zeros(1,7);
                    day_of_the_month=zeros(1,31);
                    month_of_the_year=zeros(1,12);
                    public_holiday=zeros(1,2);
                    avg_temp = 0;
                    avg_wind_speed = 0;
                    avg_humidity = 0;

                    cluster_row_index= find(Cluster_Column==j);
                    cluster_length= length(cluster_row_index);
                    if(cluster_length~=0)
                        temp_matrix=Data1(cluster_row_index,:);     

                        cluster_Sizes(crnt_cluster_ID,1) = cluster_length;
                        for k=1:cluster_length
                            public_holiday(1,(temp_matrix(k,3)+1))=public_holiday(1,(temp_matrix(k,3)+1))+1;
                            hour_of_the_day(1,(temp_matrix(k,9)+1))= hour_of_the_day(1,(temp_matrix(k,9)+1))+1;   
                            day_of_the_month(1,(temp_matrix(k,10)))= day_of_the_month(1,(temp_matrix(k,10)))+1; 
                            day_of_the_week(1,(temp_matrix(k,11)))= day_of_the_week(1,(temp_matrix(k,11)))+1;
                            month_of_the_year(1,(temp_matrix(k,12)))= month_of_the_year(1,(temp_matrix(k,12)))+1;
                            avg_temp = avg_temp + temp_matrix(k,5);
                            avg_wind_speed = avg_wind_speed +  temp_matrix(k,7);
                            avg_humidity = avg_humidity +  temp_matrix(k,8);
                        end
                        hour_of_the_day=hour_of_the_day/cluster_length;
                        day_of_the_week=day_of_the_week/cluster_length;
                        day_of_the_month=day_of_the_month/cluster_length;
                        month_of_the_year=month_of_the_year/cluster_length;
                        public_holiday=public_holiday/cluster_length;
                        avg_temp = avg_temp/cluster_length;
                        avg_wind_speed = avg_wind_speed/cluster_length;
                        avg_humidity = avg_humidity/cluster_length;


                        cluster_encoding(crnt_cluster_ID,:)= [hour_of_the_day day_of_the_week day_of_the_month month_of_the_year public_holiday avg_temp avg_wind_speed avg_humidity];


                        cluster_encoding(crnt_cluster_ID,:)=cluster_encoding(crnt_cluster_ID,:).^1;

                        crnt_cluster_ID = crnt_cluster_ID+1;
                    end
                end



                num_Sim_Clusters = zeros(total_predicted_rows,1);
                num_Sim_Clusters_by_param = zeros(total_predicted_rows,5);
                predicted_Load_Matrix = zeros(total_predicted_rows,length(Original_Dataset(1,:)));
                test_Load_Matrix = zeros(total_predicted_rows,length(Original_Dataset(1,:)));
                train_Load_Matrix = Data;

                tic
                for t=0:(total_predicted_rows-1)

                    hour_array=zeros(1,24);  
                    week_array=zeros(1,7);
                    month_day_array=zeros(1,31);
                    public_array=zeros(1,2);
                    month_array=zeros(1,12);

                    Predicted_Vector = Data_check(Predicted_Row+t,:);

                    hour_day=Predicted_Vector(9);
                    week_day=Predicted_Vector(11); %saturday
                    month_day=Predicted_Vector(10);
                    public_day=Predicted_Vector(3);
                    month_year=Predicted_Vector(12);
                    pred_temp = Predicted_Vector(5);
                    pred_wind_speed = Predicted_Vector(7);
                    pred_humidity = Predicted_Vector(8);


                    if month_year==1
                        month_str='Jan';
                    elseif month_year==2
                        month_str='Feb';
                    elseif month_year==3
                        month_str='Mar';
                    elseif month_year==4
                        month_str='April';
                    elseif month_year==5
                        month_str='May';
                    elseif month_year==6
                        month_str='June';
                    elseif month_year==7
                        month_str='July';
                    elseif month_year==8
                        month_str='Aug';
                    elseif month_year==9
                        month_str='Sept';
                    elseif month_year==10
                        month_str='Oct';
                    elseif month_year==11
                        month_str='Nov';
                    elseif month_year==12
                        month_str='Dec';
                    end

                    if week_day==1
                        week_str='Mon';
                    elseif week_day==2
                        week_str='Tue';
                    elseif week_day==3
                        week_str='Wed';
                    elseif week_day==4
                        week_str='Thurs';
                    elseif week_day==5
                        week_str='Fri';
                    elseif week_day==6
                        week_str='Sat';
                    elseif week_day==7
                        week_str='Sun';
                    end

                    if ((hour_day+1))<13
                        hour_str = 'AM';
                    else
                        hour_str = 'PM';
                    end
                    pred_time_detail = strcat(week_str,{' '},num2str(month_day),'/',month_str,...
                    {' '},num2str(hour_day+1),hour_str);
                    pred_time_detail_Times = {pred_time_detail};

                    hour_array(1,hour_day+1)=1;
                    week_array(1,week_day)=1;
                    month_day_array(1,month_day)=1;
                    public_array(1,public_day+1)=1;
                    month_array(1,month_year)=1;

                    one_hot_vector=[hour_array week_array month_day_array month_array public_array pred_temp pred_wind_speed pred_humidity];

                    % Computing Difference (Distance) between each time variable of one hot encoding vector
                    % with each cluster name time variable vector separately. Then we subtract
                    % it with 10 to compute similarity from distance

                    % Each Time vector is assigned weight to optimize the results

                    dist_hour=zeros(1,non_empty_clusters);
                    dist_week_day=zeros(1,non_empty_clusters);
                    dist_date=zeros(1,non_empty_clusters);
                    dist_month=zeros(1,non_empty_clusters);
                    dist_public=zeros(1,non_empty_clusters);

                    dist_by_param = zeros(non_empty_clusters,8);
                    sim_by_param = zeros(non_empty_clusters,8);

        %             p_of_LP_Norm = 1;
                    for i=1:non_empty_clusters
                        sim_by_param(i,1) = cluster_encoding(i,hour_day+1) + 0.8*cluster_encoding(i,mod(hour_day+1,24)+1) + 0.8*cluster_encoding(i,mod(hour_day-1,24)+1) + 0.5*cluster_encoding(i,mod(hour_day+2,24)+1)+ 0.5*cluster_encoding(i,mod(hour_day-2,24)+1) + 0.25*cluster_encoding(i,mod(hour_day-3,24)+1) + 0.25*cluster_encoding(i,mod(hour_day+3,24)+1) ;
                        if (week_day>=1 && week_day<=4)
                            sim_by_param(i,2) = cluster_encoding(i,25) + cluster_encoding(i,26) + cluster_encoding(i,27) + cluster_encoding(i,28);    
                        elseif (week_day==6 || week_day==7)
                            sim_by_param(i,2) = cluster_encoding(i,30) + cluster_encoding(i,31);
                        else
                            sim_by_param(i,2) = cluster_encoding(i,29) + .4*cluster_encoding(i,30) + .4*cluster_encoding(i,28)+.4*cluster_encoding(i,27);                       
                        end
                        sim_by_param(i,3) = cluster_encoding(i,32+month_day-1) + .5*cluster_encoding(i,32+mod(month_day-1+1,31)) + .5*cluster_encoding(i,32+mod(month_day-1-1,31)) + .25*cluster_encoding(i,32+mod(month_day-1+2,31))+ .25*cluster_encoding(i,32+mod(month_day-1-2,31));
                        sim_by_param(i,4) = cluster_encoding(i,63+month_year-1) + .5*cluster_encoding(i,63+mod(month_year-1+1,12)) + .5*cluster_encoding(i,63+mod(month_year-1-1,12)) + .3*cluster_encoding(i,63+mod(month_year-1+2,12))+ .3*cluster_encoding(i,63+mod(month_year-1-2,12));
                        sim_by_param(i,5) = cluster_encoding(i,75+public_day);
                        sim_by_param(i,6) = cluster_encoding(i,77);
                        sim_by_param(i,7) = cluster_encoding(i,78);
                        sim_by_param(i,8) = cluster_encoding(i,79);
                    end %    end of  for i=1:non_empty_clusters

                    hour_Weight = 0.3;
                    week_Weight = 0.1;
                    month_day_Weight = 0.06;
                    month_Weight = 0.47;
                    public_Weight = 0.04;
                    public_temp = 0.01;
                    public_wind = 0.01;
                    public_humid = 0.01;


                    sum_Similarity  = hour_Weight*sim_by_param(:,1) + week_Weight * sim_by_param(:,2) + ...
                        month_day_Weight * sim_by_param(:,3) ...
                        + month_Weight * sim_by_param(:,4) + public_Weight * sim_by_param(:,5)...
                     + public_temp * sim_by_param(:,6) + public_wind * sim_by_param(:,7)...
                      + public_humid * sim_by_param(:,8);

                    k_val=2;
                    [val ind] = sort(sum_Similarity,'descend');
                    top_k=val(1:k_val);
                    sum_Similarity(:,1)=0;
                    sum_Similarity(ind(1:k_val),1) = val(1:k_val);

                    num_Sim_Clusters(t+1) = sum(sum_Similarity>=1.2);

                    pow = 5;
                    r=1;
                    for i=1:length(sum_Similarity)
                        if(sum_Similarity(i) ~=0)
                            Cluster_i_Matrix = train_Load_Matrix(Cluster_Column==(i),:);
                            if(~isempty(Cluster_i_Matrix))
                                if length(Cluster_i_Matrix(:,1))==1
                                    predicted_Load_Matrix(t+1,:) = predicted_Load_Matrix(t+1,:)+ (sum_Similarity(i).^pow* Cluster_i_Matrix);
                                else
                                    predicted_Load_Matrix(t+1,:) = predicted_Load_Matrix(t+1,:)+ (sum_Similarity(i).^pow* (median(Cluster_i_Matrix.^(1/r)).^r));
                                end
                            end
                        end
                     end %for i=1:length(sim_over_all) ends here

                    predicted_Load_Matrix(t+1,:) = predicted_Load_Matrix(t+1,:)/ sum(sum_Similarity.^pow);
                end

                test_Load_Matrix = Original_Dataset(Predicted_Row:Predicted_Row+total_predicted_rows-1,:);
                POWER;
                test_Load_Matrix = test_Load_Matrix.^POWER;
                predicted_Load_Matrix = predicted_Load_Matrix.^POWER;
                error_predicted_Actual = (predicted_Load_Matrix) - (test_Load_Matrix);
                absError_predicted_Actual = abs(error_predicted_Actual);

                err{z*2-1,1}=test_Load_Matrix;
                err{z*2,1}=predicted_Load_Matrix;

                row_sum_predicted = sum(predicted_Load_Matrix,2);
                row_sum_actual = sum(test_Load_Matrix,2);
                abs_difference = (row_sum_predicted) - (row_sum_actual);
                error_abs = abs(abs_difference);
                error_percent = (error_abs./row_sum_actual)*100;
                trimmean_MAPE = trimmean(error_percent,10)
                CalcPerf(row_sum_actual,error_abs);
                trimmean_MAE = trimmean(trimmean(absError_predicted_Actual,10),10)


                CalcPerf(test_Load_Matrix,predicted_Load_Matrix)
                MAE = trimmean(trimmean(predicted_Load_Matrix,10),10)

    %             dlmwrite(strcat('D:\Pasha\Soft Load Shedding\Results\new\',dataset_name_string,'_Actual_values_power_',num2str(POWER),'_dim_',num2str(dimensions),'_clusters_',num2str(Cluster_Range),'.csv'),test_Load_Matrix);
    %             dlmwrite(strcat('D:\Pasha\Soft Load Shedding\Results\new\',dataset_name_string,'_Predicted_values_power_',num2str(POWER),'_dim_',num2str(dimensions),'_clusters_',num2str(Cluster_Range),'.csv'),predicted_Load_Matrix);


    %             fprintf('\ncust-wise-MAE = %3.2f %3.2f  %3.2f %3.2f \n',mean(absError_predicted_Actual(1,:)),mean(absError_predicted_Actual(200,:)),mean(absError_predicted_Actual(2400,:)),mean(absError_predicted_Actual(3100,:)))
        %         fprintf('time-wise-MAE = %3.2f %3.2f  %3.2f %3.2f \n',mean(absError_predicted_Actual(:,1)),mean(absError_predicted_Actual(:,32)),mean(absError_predicted_Actual(:,59)),mean(absError_predicted_Actual(:,73)))

        %      [mean(absError_predicted_Actual(:,1)) mean(absError_predicted_Actual(:,7)) mean(absError_predicted_Actual(:,19)) mean(absError_predicted_Actual(:,73))]

                percentError_predicted_Actual = (absError_predicted_Actual./test_Load_Matrix)*100;

                Nans_in_Predictions = isnan(percentError_predicted_Actual);
                Infs_in_Predictions = isinf(percentError_predicted_Actual);
                Nans_or_Inf_Predictions = Nans_in_Predictions+Infs_in_Predictions;
                num_Nans_or_Inf_columnsWise = sum(Nans_or_Inf_Predictions);
                num_Nans_or_Inf_rowWise = sum(Nans_or_Inf_Predictions,2);

        %         InfIndices = (percentError_predicted_Actual==Inf);
        %         percentError_predicted_Actual = percentError_predicted_Actual(percentError_predicted_Actual~=Inf);

        %         sum( percentError_predicted_Actual(~isnan(percentError_predicted_Actual) && ~isinf(percentError_predicted_Actual)));

        %         percentError_predicted_Actual(percentError_predicted_Actual==Inf) = 0;
                percentError_predicted_Actual(isnan(percentError_predicted_Actual))=0;
                percentError_predicted_Actual(isinf(percentError_predicted_Actual))=0;

        %         avg_predicitonError_ColSum = trimmean(percentError_predicted_Actual,10);
        %         avg_predicitonError_RowSum = trimmean(percentError_predicted_Actual,10,2);

                avg_predicitonError_ColSum = mean(percentError_predicted_Actual);
                avg_predicitonError_RowSum = mean(percentError_predicted_Actual,2);

                round(quantile(avg_predicitonError_RowSum,10),2);

                round(quantile(avg_predicitonError_ColSum,10),2);

        %         avg_predicitonError_RowSum = predicitonError_RowSum./(length(test_Load_Matrix(1,:))-num_Nans_or_Inf_rowWise);
        %         avg_predicitonError_ColSum = predicitonError_ColSum./(length(test_Load_Matrix(:,1))-num_Nans_or_Inf_columnsWise);

%                 fprintf('\nPer Time (row) Average MAPE EXACT = %3.2f   STD MAPE EXACT = %3.2f   Median MAPE EXACT = %3.2f\n', mean(avg_predicitonError_RowSum),std(avg_predicitonError_RowSum),median(avg_predicitonError_RowSum));
%                 fprintf('\nPer Customer (col) Average MAPE EXACT = %3.2f   STD MAPE EXACT = %3.2f   Median MAPE EXACT = %3.2f\n', mean(avg_predicitonError_ColSum),std(avg_predicitonError_ColSum),median(avg_predicitonError_ColSum));

                mape_err = mean(avg_predicitonError_ColSum);
                aa_1 = CalcPerf(test_Load_Matrix,predicted_Load_Matrix);
                rmse_err = aa_1.RMSE;
                mae_err = mean(mean(absError_predicted_Actual));


                    if(mape_err<min_mape)
                        min_mape = mape_err
                    end

                    if(rmse_err<min_rmse)
                        min_rmse = rmse_err
                    end

                    if(mae_err<min_mae)
                        min_mae = mae_err
                    end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%% SMAPE Error Measure (Starts Here)%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [row_check,col_check]=size(test_Load_Matrix);
                sym_MAPE=zeros(row_check,col_check);
                for q=1:row_check
                    for w=1:col_check
                sym_MAPE(q,w)= abs(test_Load_Matrix(q,w)-predicted_Load_Matrix(q,w))/...
                    (0.5*(test_Load_Matrix(q,w)+predicted_Load_Matrix(q,w)));
                    end
                end
                SMAPE = mean(mean(sym_MAPE))*100;
                smape_matrix(row_temp_2+1,col_temp+1) = SMAPE;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%% SMAPE Error Measure (Ends Here)%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                mape_matrix(row_temp_2+1,col_temp+1) = mean(avg_predicitonError_ColSum);

                count_holder = 17544;
                count_holder_2 = count_holder/length(Original_Dataset_initial(:,1));
                aa_1 = CalcPerf(test_Load_Matrix,predicted_Load_Matrix);
                rmse_matrix(row_temp_2+1,col_temp+1) = aa_1.RMSE;

                if z<=6
                mape_matrix(z+1,1) = length(Original_Dataset_initial(1,:));
                smape_matrix(z+1,1) = length(Original_Dataset_initial(1,:));
                rmse_matrix(z+1,1) = length(Original_Dataset_initial(1,:));

                end
                row_temp_2 = row_temp_2+1;
                row_temp = row_temp+1;
                if z==6
                    col_temp = col_temp+1;
                    row_temp_2 = 1;
                    mape_matrix(1,2) = count_holder_2;
                    smape_matrix(1,2) = count_holder_2;
                    rmse_matrix(1,2) = count_holder_2;
                elseif z==12
                    col_temp = col_temp+1;
                    row_temp_2 = 1;
                     mape_matrix(1,3) = count_holder_2;
                     smape_matrix(1,3) = count_holder_2;
                    rmse_matrix(1,3) = count_holder_2;
                    elseif z==18
                    col_temp = col_temp+1;
                    row_temp_2 = 1;
                     mape_matrix(1,4) = count_holder_2;
                     smape_matrix(1,4) = count_holder_2;
                    rmse_matrix(1,4) = count_holder_2;
                    elseif z==24
                    col_temp = col_temp+1;
                    row_temp_2 = 1;
                     mape_matrix(1,5) = count_holder_2;
                     smape_matrix(1,5) = count_holder_2;
                    rmse_matrix(1,5) = count_holder_2;
                    elseif z==30
                    col_temp = col_temp+1;
                    row_temp_2 = 1;
                     mape_matrix(1,6) = count_holder_2;
                     smape_matrix(1,6) = count_holder_2;
                    rmse_matrix(1,6) = count_holder_2;
                end


        toc
    end
end %iterr loop ends here

% Sort based on Column
 [~,idx] = sort(mape_matrix(:,1)); % sort just the first column
sortmape = mape_matrix(idx,:);   % sort the whole matrix using the sort indices

 [~,idx] = sort(smape_matrix(:,1)); % sort just the first column
sortsmape = smape_matrix(idx,:);   % sort the whole matrix using the sort indices


 [~,idx] = sort(rmse_matrix(:,1)); % sort just the first column
sortrmse = rmse_matrix(idx,:);   % sort the whole matrix using the sort indices


% Sort based on row
 [~,idx] = sort(sortmape(1,:)); % sort just the first column
sortmape_final = sortmape(:,idx);   % sort the whole matrix using the sort indices

 [~,idx] = sort(sortsmape(1,:)); % sort just the first column
sortsmape_final = sortsmape(:,idx);   % sort the whole matrix using the sort indices

 [~,idx] = sort(sortrmse(1,:)); % sort just the first column
sortrmse_final = sortrmse(:,idx);   % sort the whole matrix using the sort indices

%mape figure
names_mape = {num2str(sortmape_final(1,2)),num2str(sortmape_final(1,3)),num2str(sortmape_final(1,4)),...
    num2str(sortmape_final(1,5)),num2str(sortmape_final(1,6))};
X = [1:5];
figure
% plot(sortmape_final(2:7,2:7));
fig1=plot(X,sortmape_final(2,2:6),':r*',X,sortmape_final(3,2:6),'--kp',...
    X,sortmape_final(4,2:6),'--mo',X,sortmape_final(5,2:6),':+',...
    X,sortmape_final(6,2:6),'-.gs',X,sortmape_final(7,2:6),'-bs',...
    'LineWidth',1.5,'MarkerSize',8);
% title('(Sweden Dataset) Effect of Hours and Customers aggregation on MAPE');
xlabel('Increasing Hours Sum') ;
ylabel('MAPE') 
set(gca,'xtick',[1:6],'xticklabel',names_mape)
set(gcf,'color','white','InvertHardCopy','off');
set(gcf, 'Position', get(0, 'Screensize'));
set(gca,'box','on','LineWidth',2.5,'FontName','Helvetica','FontSize',16,'TickLength',[0.005 0.05],'PlotBoxAspectRatio',[1 0.85 1],...
        'xscale','linear','xgrid','off','ygrid','on','YMinorTick','off','XMinorTick','off',...
        'XColor','k','YColor','k')
%         ylim([0 100]);
legend(strcat('Clust\_num: ',num2str(sortmape_final(2,1))),strcat('Clust\_num: ',num2str(sortmape_final(3,1))),...
    strcat('Clust\_num: ',num2str(sortmape_final(4,1))),...
    strcat('Clust\_num: ',num2str(sortmape_final(5,1))),strcat('Clust\_num: ',num2str(sortmape_final(6,1)))...
    ,strcat('Clust\_num: ',num2str(sortmape_final(7,1))));    
    legend('boxoff');
    lgd=legend;
    lgd.FontSize = 12;



% SMAPE figure
names_mape = {num2str(sortsmape_final(1,2)),num2str(sortsmape_final(1,3)),num2str(sortsmape_final(1,4)),...
    num2str(sortsmape_final(1,5)),num2str(sortsmape_final(1,6))};
figure
% plot(sortmape_final(2:7,2:7));
fig1=plot(X,sortsmape_final(2,2:6),':r*',X,sortsmape_final(3,2:6),'--kp',...
    X,sortsmape_final(4,2:6),'--mo',X,sortsmape_final(5,2:6),':+',...
    X,sortsmape_final(6,2:6),'-.gs',X,sortsmape_final(7,2:6),'-bs',...
    'LineWidth',1.5,'MarkerSize',8);
% title('(Sweden Dataset) Effect of Hours and Customers aggregation on SMAPE');
xlabel('Hours Aggregation') ;
ylabel('SMAPE') 
set(gca,'xtick',[1:6],'xticklabel',names_mape)
set(gcf,'color','white','InvertHardCopy','off');
set(gcf, 'Position', get(0, 'Screensize'));
set(gca,'box','on','LineWidth',2.5,'FontName','Helvetica','FontSize',16,'TickLength',[0.005 0.05],'PlotBoxAspectRatio',[1 0.85 1],...
        'xscale','linear','xgrid','off','ygrid','on','YMinorTick','off','XMinorTick','off',...
        'XColor','k','YColor','k')
%         ylim([0 100]);
legend(strcat('Clust\_num: ',num2str(sortsmape_final(2,1))),strcat('Clust\_num: ',num2str(sortsmape_final(3,1))),...
    strcat('Clust\_num: ',num2str(sortsmape_final(4,1))),...
    strcat('Clust\_num: ',num2str(sortsmape_final(5,1))),strcat('Clust\_num: ',num2str(sortsmape_final(6,1)))...
    ,strcat('Clust\_num: ',num2str(sortsmape_final(7,1))));    
    legend('boxoff');
    lgd=legend;
    lgd.FontSize = 12;


%RMSE figure
names = {num2str(sortrmse_final(1,2)),num2str(sortrmse_final(1,3)),num2str(sortrmse_final(1,4)),...
    num2str(sortrmse_final(1,5)),num2str(sortrmse_final(1,6))};
figure
% plot(sortmape_final(2:7,2:7));
fig1=plot(X,sortrmse_final(2,2:6),':r*',X,sortrmse_final(3,2:6),'--kp',...
    X,sortrmse_final(4,2:6),'--mo',X,sortrmse_final(5,2:6),':+',...
    X,sortrmse_final(6,2:6),'-.gs',X,sortrmse_final(7,2:6),'-bs',...
    'LineWidth',1.5,'MarkerSize',8);
% title('(Sweden Dataset) Effect of Hours and Customers aggregation on RMSE');
xlabel('Hours Aggregation') ;
ylabel('RMSE') 
set(gca,'xtick',[1:6],'xticklabel',names_mape)
set(gcf,'color','white','InvertHardCopy','off');
set(gcf, 'Position', get(0, 'Screensize'));
set(gca,'box','on','LineWidth',2.5,'FontName','Helvetica','FontSize',16,'TickLength',[0.005 0.05],'PlotBoxAspectRatio',[1 0.85 1],...
        'xscale','linear','xgrid','off','ygrid','on','YMinorTick','off','XMinorTick','off',...
        'XColor','k','YColor','k')
%         ylim([0 100]);
legend(strcat('Clust\_num: ',num2str(sortrmse_final(2,1))),strcat('Clust\_num: ',num2str(sortrmse_final(3,1))),...
    strcat('Clust\_num: ',num2str(sortrmse_final(4,1))),...
    strcat('Clust\_num: ',num2str(sortrmse_final(5,1))),strcat('Clust\_num: ',num2str(sortrmse_final(6,1)))...
    ,strcat('Clust\_num: ',num2str(sortrmse_final(7,1))));    
    legend('boxoff');
    lgd=legend;
    lgd.FontSize = 12;


csvwrite('csv results\Sweden_MAPE_After_Normalized_temperature_in_SVD.csv',sortmape_final);
csvwrite('csv results\Sweden_SMAPE_After_Normalized_temperature_in_SVD.csv',sortsmape_final);
csvwrite('csv results\Sweden_RMSE_After_Normalized_temperature_in_SVD.csv',sortrmse_final);

