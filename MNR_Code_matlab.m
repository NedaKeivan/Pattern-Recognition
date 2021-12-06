clc
clear 
close all
 
%%Train MNR 
% First of all we need to split dataset to training and test set

DataSet = xlsread('iris_dataset_xls');
IrisReduced = DataSet(:, 3:end);

% Add a column of ones to X
X = [IrisReduced , ones(size(IrisReduced,1),1)];
Trainingset1 = X(1:40, :);
Testset1 = X(41:50, :);
Trainingset2 = X(51:90, :);
Testset2 = X(91:100, :);
Trainingset3 = X(101:140, :);
Testset3 = X(141:150, :);
% consider the alphamax equal to 300
alphaMax = 300;

X = [Trainingset1;Trainingset2;Trainingset3];
Xtest = [Testset1;Testset2;Testset3];

% one-hot-encoding
t1 = [ones(40,1) ; zeros(80,1)];
t2 = [zeros(40,1); ones(40,1); zeros(40,1)];
t3 = [zeros(80,1); ones(40,1)];
T = [t1, t2, t3];
W1 = zeros(2,1); 
W2 = zeros(2,1); 
W3 = zeros(2,1);
b = zeros(1,3);
Wt = [W1, W2, W3];
W = [Wt ; b];
test =[];

% Training the MNR model 
 while(true) 
    alpha = alphaMax; %alphaMax is 300
    Num_trial = 0;
    % it is multiply of weight and Training sets
    H = X * W; 
    H_Max = max(H,[],2);
    % to avoid having nan's result we have the following subtraction of all which is less or equal to zero
    robust = H - H_Max;
    S = exp(robust) ./ repmat(sum(exp(robust),2), 1, 3);
    d_ACE = (1/120) * X' * (S - T); 
    idx0 = find(S==0);
    S(idx0) = 1e-30;
    % Average Cross Entropy
    ACE = (1/120) * (sum(-sum(T .* log(S), 2)));
    test = [test;ACE];
    NORM = norm(d_ACE,1);
    if(NORM< 0.01)
      break;
    end 
    while(true)
    W_t = W - (alpha * d_ACE);
    H2 = X * W_t;
    H_Max2 = max(H2,[],2);
    robust2 = H2 - H_Max2;
    S2 = exp(robust2) ./ repmat(sum(exp(robust2),2), 1, 3);
    idxs0 = find(S2==0);
    S2(idxs0) = 1e-30;
    ACE2 = (1/120) * (sum(-sum(T .* log(S2), 2)));
        if (ACE2 > ACE & Num_trial <300)
            alpha = alpha * rand(1);
        else
         % Use trial point as a new point and exit 
         W = W_t;
         break;
        end
        Num_trial = Num_trial + 1;
    end
 end
[val,loc] = max(S2');

% Show the obtained decision regions
Class1=[]; 
Class2=[]; 
Class3=[];
idx1 = find(loc == 1);
idx2 = find(loc == 2);
idx3 = find(loc == 3);
Class1 = X(idx1, :);
Class2 = X(idx2, :);
Class3 = X(idx3, :);
figure()
hold on
scatter(Class1(:,1),Class1(:,2),50,'filled')
scatter(Class2(:,1),Class2(:,2),50,'filled')
scatter(Class3(:,1),Class3(:,2),50,'filled')
xlabel('Petal Length','FontSize',20,'FontWeight','bold');
ylabel('Petal Width','FontSize',20,'FontWeight','bold');
title('Decision regions for trained MNR','FontSize',24);

% consider Lambda = 0 and find MNR misclassification rate obtained for this
% Lambda
Error_H = [];
Htest = Xtest * W;
HtestMax = max(Htest,[],2);
robusttest = Htest - HtestMax;
Softmaxtest = exp(robusttest) ./ repmat(sum(exp(robusttest),2), 1, 3);
[valtest,loctest] = max(Softmaxtest');
err1 = sum (loctest(1:10) ~= 1);
err2 = sum (loctest(11:20) ~= 2);
err3 = sum (loctest(21:30) ~=3);
M1 = sum([err1, err2, err3])/30;
Error_H = [Error_H, M1];
j = 1:1:size(test);
figure()
plot (test,j);
xlabel('iteration', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Average-Cross-Entropy', 'FontSize', 20, 'FontWeight', 'bold');
title('ACE at each iteration ','FontSize',24);
Training_Error = [];
% Training error
[valtrain1,loctrain1] = max(S');
errortrain1 = sum (loctrain1(1:40) ~= 1);
errortrain2 = sum (loctrain1(41:80) ~= 2);
errortrain3 = sum (loctrain1(81:120) ~= 3);
Mtrain1 = sum([errortrain1, errortrain2, errortrain3])/120;
Training_Error = [Training_Error, Mtrain1];


%% Regularized MNR
DataSet = xlsread('iris_dataset_xls');
IrisReduced = DataSet(:, 3:end);
alphaMax = 300;
Error_H = [0];
Training_Error3 =[0.05];
% one-hot-encoding
t1 = [ones(40,1) ; zeros(80,1)];
t2 = [zeros(40,1); ones(40,1); zeros(40,1)];
t3 = [zeros(80,1); ones(40,1)];
T = [t1, t2, t3];
phi = [IrisReduced(:,1), IrisReduced(:,2), IrisReduced(:,1).*IrisReduced(:,2), IrisReduced(:,1).^2, IrisReduced(:,2).^2,  ones(150,1)];
Trainingset1_phi = phi(1:40, :);
Testset1_phi = phi(41:50, :);
Trainingset2_phi = phi(51:90, :);
Testset2_phi = phi(91:100, :);
Trainingset3_phi = phi(101:140, :);
Testset3_phi = phi(141:150, :);
phi = [Trainingset1_phi ; Trainingset2_phi ; Trainingset3_phi];
Xtest_phi = [Testset1_phi ; Testset2_phi ; Testset3_phi];

 for k = -7:0.2:7
    Lambda = 10^k; 
    W1 = zeros((size(phi,2) -1),1); 
    W2 = zeros((size(phi,2) -1),1); 
    W3 = zeros((size(phi,2) -1),1);
    b = zeros(1,3);
    Wt = [W1, W2, W3];
    W = [Wt ; b];
    % Training
    test3 =[];
    while (true)
        alpha = alphaMax;
        %value of learning rate and maximum learning rate
        H = phi * W;
        H_Max = max(H,[],2);
        robust = H - H_Max;
        %softmax activation function
        SR = exp(robust) ./ repmat(sum(exp(robust),2), 1, 3);
        idxSR = find(SR==0);
        SR(idxSR) = 1e-30;
        d_RACE = (1/120) * phi' * (SR - T) + (Lambda .* [Wt; zeros(1,3)]);
        RACE = ((1/120) * (sum(-sum(T .* log(SR), 2)))) + (Lambda/2)* trace(Wt'* Wt);
        test3 = [test3;RACE];
        NORM = norm(d_RACE,1);
        if(NORM< 0.4)
             break;
        end 
        W_t2 = W - (alpha * d_RACE);
        Wt_trial2 = W(1:(size(phi,2) -1), :);       
        H2 = phi * W_t2;
        H_Max2 = max(H2,[],2);
        robust2 = H2 - H_Max2;
        S2 = exp(robust2) ./ repmat(sum(exp(robust2),2), 1, 3);
        idx2 = find(S2==0);
        S2(idx2) = 1e-30;
        RACE2 = ((1/120) * (sum(-sum(T .* log(S2), 2))))+ (Lambda/2)* trace(Wt_trial2'* Wt_trial2);
        j= 0;
        while(RACE2 > RACE  & j<300)
            alpha = alpha * rand(1);
            W_t = W - (alpha * d_RACE);
            Wt_trial = W_t(1:(size(phi,2) -1), :);       
            H2 = phi * W_t;
            H_Max2 = max(H2,[],2);
            robust2 = H2 - H_Max2;
            S2 = exp(robust2) ./ repmat(sum(exp(robust2),2), 1, 3);
            idx = find(S2 == 0);
            S2(idx) = 1e-30;
            RACE2 = ((1/120) * (sum(-sum(T .* log(S2), 2))))+ (Lambda/2)* trace(Wt_trial'* Wt_trial);  
            j= j+1;
        end
         W = W_t;
         Wt = W(1:(size(phi,2) -1), :);       
    end
        % MNR misclassification rate for different Lambda 
        Htest = Xtest_phi * W;
        HtestMax = max(Htest,[],2);
        robusttest = Htest - HtestMax;
        Softmaxtest = exp(robusttest) ./ repmat(sum(exp(robusttest),2), 1, 3);
        [valtest,loctest] = max(Softmaxtest');
        err1 = sum (loctest(1:10) ~= 1);
        err2 = sum (loctest(11:20) ~= 2);
        err3 = sum (loctest(21:30) ~= 3);
        M1 = sum([err1, err2 err3])/30;
        Error_H = [Error_H, M1];
        % Training error
        [valtrain3,loctrain3] = max(SR');
        errortrain1 = sum (loctrain3(1:40) ~= 1);
        errortrain2 = sum (loctrain3(41:80) ~= 2);
        errortrain3 = sum (loctrain3(81:120) ~= 3);
        Mtrain3 = sum([errortrain1, errortrain2, errortrain3])/120;
        Training_Error3 = [Training_Error3; Mtrain3];
 end
[val3,loc3] = max(SR');
lam = 10.^(-7:0.2:7);
lam = [0,lam];
figure
plot(lam, Error_H)
xlabel('Lambda', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Hold-out set error', 'FontSize', 20, 'FontWeight', 'bold');
title('Misclassification rate on the hold-out set','FontSize',20);
% Training error
j = 1:1:size(Training_Error3);
figure()
plot (j,Training_Error3);
xlabel('iteration', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Training error', 'FontSize', 20, 'FontWeight', 'bold');
title('Misclassification rate on training set ','FontSize',20);
k = 1:1:size(test3);
figure()
plot (test3,k);
xlabel('iteration', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Average-Cross-Entropy at each iteration', 'FontSize', 20, 'FontWeight', 'bold');
title('Regularized Average-Cross-Entropy','FontSize',24);