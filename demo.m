%demo code of the linear and kernel kNN hashing

clc;
clear all;
close all;

addpath(genpath('./'));
load ./data/Mnist4k.mat%a mini dataset

%% kNNH
opts.theta_scl = 0.8; %may not be optimal
opts.appr_type = {'tanh'};
opts.norm_type = 'avg_norm';
opts.nc = 3; %number of prototypes each class
opts.K = 24; %K bit hash codes
model = kNNH(data.Xtrain, data.ytrain, opts);
Bretri = gen_bits(data.Xretri, model, 'linear');
Btest = gen_bits(data.Xtest, model, 'linear');
D = hammDist_mex(Bretri, Btest);
[~, IX] = sort(D, 1, 'ascend');
map1 = MAP(data.Yretri, data.Ytest, IX);
map1

%% k2NNH
opts.theta_scl = 0.2; %may not be optimal
opts.gamma_scl = 0.3; %may not be optimal
opts.appr_type = {'tanh'};
opts.norm_type = 'avg_norm';
opts.nc = 3; %number of prototypes each class
opts.K = 24; %K bit hash codes
model = k2NNH(data.Xtrain, data.ytrain, opts);
Bretri = gen_bits(data.Xretri, model, 'kernel');
Btest = gen_bits(data.Xtest, model, 'kernel');
D = hammDist_mex(Bretri, Btest);
[~, IX] = sort(D, 1, 'ascend');
map2 = MAP(data.Yretri, data.Ytest, IX);
map2

%% SDH
model = SDHtrain(data.Xtrain', data.ytrain', 24, 30*10, 0.8); %300 anchors in total
Bretri = SDHtest(data.Xretri', model);
Btest = SDHtest(data.Xtest', model);
D = hammDist_mex(Bretri, Btest);
[~, IX] = sort(D, 1, 'ascend');
map3 = MAP(data.Yretri, data.Ytest, IX);
map3
