clc;
clear;

% Set up parameters
num_curves = 2000;     % if we increase the number of curves, error decreases (5000, 10 000, etc...)
npontosmax=100;
tspan = linspace(0, 60, npontosmax); % Time span for simulation  %60 seg
tconv=linspace(min(tspan)*2,max(tspan)*2,length(tspan)*2-1);   % time for convolution

%beggining %(rows, columns)
% ap_curves = zeros(length(tspan), num_curves);  
% h_curves = zeros(length(tspan), num_curves);  
% ah_curves = zeros(length(tspan), num_curves);
% ah_curves_conv = zeros(length(tconv), num_curves);

%inicialization vectors for the table
vec_num=[];
vec_ap=[];
vec_h=[];
vec_ahconv=[];

%parameters quantification
vec_hef=[];


for i = 1:num_curves

    vec_num=[vec_num i];

    alpha=0.5+(0.9-0.5)*rand;
    beta=0.5+(0.7-0.5)*rand;
    gamma=0.1+(0.4-0.1)*rand;

    while gamma>beta
        gamma=0.1+(0.4-0.1)*rand;
    end

    k1=0.12+(0.20-0.12)*rand;     %limits for k1 and k2 
    k2=0.12+(0.20-0.12)*rand;

    % plasmatic activity curve
    ap_curves (:,i)= apfunction(tspan, k1, alpha);    
    vec_ap=[vec_ap ap_curves(:,i)];

    % hepatic activity curve
    ah_curves (:,i)= ahfunction(tspan, k1, k2, beta, gamma);      %curva ah theoretical

    %hepatic retention curve
    h_curves (:,i)= hfunction(tspan,k1,k2, alpha, beta, gamma);  
    vec_h=[vec_h h_curves(:,i)];
   
    %hepatic activity curve CONVOLUTION
    ah_curves_conv (:,i)= conv(ap_curves(:,i),h_curves(:,i),'full');  %curve ah from convolution
    vec_ahconv=[vec_ahconv ah_curves_conv(1:npontosmax,i)];




    %HEF  (from h(t))
    A=gamma/alpha;    
    B=(beta/alpha)*k1;
    hef=(B/A)*100;         %% B/A!! not A/B !!! 
    if hef>100     % if B>A, HEF=100%
        hef=100;
    end
    vec_hef=[vec_hef hef];
    
   
end

%table with 2000 sets of corresponding curves ap, h e ah (from convolution) 
dataTable = table(vec_num', vec_ap', vec_ahconv', vec_h', vec_hef', 'VariableNames', {'Num', 'ap', 'ah_conv', 'h', 'hef(%)'});
disp(dataTable);



% MACHINE LEARNING  !!! 
% give features (ap, ah) and label (HEF) -> train model
% in the future, give features of any image and receive a
% prediction for HEF (from trained algorithm) :)


% split training set e test set

F=dataTable(:,2:3);
F1=table2array(F);

L=dataTable(:,end);     %labels (HEF)
L1=table2array(L);         % labels array (1 -> HEF)

numPartitions=5;
sumErrors=0;

for i=1:numPartitions
    cv = cvpartition(size(F1, 1), 'HoldOut', 0.3);  % 40% for testing

    trainIdx = training(cv);
    testIdx = test(cv);

    % Split the data into training and test sets
    Ftrain = F1(trainIdx, :);
    Ltrain = L1(trainIdx, :);
    
    Ftest = F1(testIdx, :);
    Ltest = L1(testIdx, :);


    svrModel = fitrsvm(Ftrain, Ltrain, 'KernelFunction', 'gaussian', 'Standardize', true);
    trainedModel = fitrsvm(Ftrain, Ltrain);
    Lpred = predict(trainedModel, Ftest);


    %metrics for evaluation!! ???
    mse = mean((Lpred - Ltest).^2);
    rmse = sqrt(mse);


    error_abs=abs(Lpred-Ltest);
    error_rel=(error_abs./Ltest)*100;     % THE POINT (./)

    error_rel_mean=mean(error_rel);
    sumErrors=sumErrors+error_rel_mean;

end

error_rel_mean_final=sumErrors/numPartitions


% Plot all curves in one plot
figure;
subplot(2,2,1)
plot(tspan, ap_curves,'red');         %curve ap rand (theoretical)
xlabel('time');
ylabel('ap activity');
ylim([0,1])

subplot(2,2,2)
plot(tspan, h_curves,'black');          %curve h rand (theor)
xlabel('time');
ylabel('h activity');
ylim([0,1])

subplot(2,2,3)
plot(tspan, ah_curves,'green');             %curve ah rand (theor)
xlabel('time');
ylabel('ah activity (theoretical curve)');
ylim([0,0.6])

subplot(2,2,4)
plot(tspan, ah_curves_conv(1:npontosmax,:),'blue');    %curve ah (from conv)
xlabel('time');
ylabel('ah activity (conv curve)');
ylim([0,1])


% maximum HEF?? (confirmation)
%max(vec_hef)


%functions of different activities

function ap = apfunction(t, k1, alpha)
    A0=1;
    ap=A0*alpha*exp(-k1.*t);
    
    %adding noise
    noise_std = 0.01; % Standard deviation of the noise
    ap = ap + noise_std * randn(size(ap));
end

function ah = ahfunction(t, k1, k2, beta, gamma)     %curva teorica
    A0=1;
    Ap=beta*(k1/(k2-k1))*A0 + gamma*A0;
    Ah=-Ap + gamma*A0;
    ah=Ap*exp(-k1.*t)+Ah*exp(-k2.*t);

    %adding noise 
    noise_std = 0.01; % Standard deviation of the noise
    ah = ah + noise_std * randn(size(ah));
end

function h = hfunction(t,k1,k2, alpha, beta, gamma)
    %A=gamma/alpha;    
    B=(beta/alpha)*k1;
    h=B.*exp(-k2.*t);

    %adding noise
    noise_std = 0.01; % Standard deviation of the noise
    h = h + noise_std * randn(size(h));

    h(1) = gamma/alpha;  %vascular peak
end


