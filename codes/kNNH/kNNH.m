function model = kNNH(X, y, opts)
%train the linear kNN hashing model
%INPUT:
%X: data matrix, each column is a data point
%y: labels, row vector with each entry in {1,2,...,C}, C is the number of
%   classes
%opts:
% -K: number of hash bits
% -nc: number of prototypes learned for each class
% -appr_type: type of approximation function for sign function
% -theta: parameter controlling the number of neighbors in defining \Pi_{ij}
% -theta_scl: scaling factor of theta, it should be given if theta is not
%             provided
% -alpha: trade-off parameter
%
%OUTPUT:
%model:
% -W: projection matrix, each column corresponds a hash function
% -Z: prototypes, each column is a prototype
% -yhat: labels of prototypes, row vector
% -theta: parameter controlling the number of neighbors in defining \Pi_{ij}
% -normlize_info: information used for normalizing the training data
% -tim: training time

%The code is implemented by Kun Ding (kding@nlpr.ia.ac.cn).

tic;

%data pre-processing
[X, normlize_info] = data_normalization(X, opts.norm_type);

[d, n] = size(X);
C = length(unique(y)); %number of classes
nc = opts.nc;
m = sum(nc)*C; %total number of prototypes
K = opts.K; %K-bit hash
if(~isfield(opts,'alpha'))
    alpha = 1e-4*n/m; %default setting of alpha
else
    alpha = opts.alpha*n/m;
end

%initialize R
Y = sparse(y, 1:n, ones(1,n), C, n); %label matrix of X
yhat = reshape(repmat(1:C, nc, 1), [], 1);
Yhat = sparse(yhat, 1:m, ones(1,m), C, m); %label matrix of Z
R = Y'*Yhat; %relation matrix between training samples and anchors
clear Yhat;

%initialize Z by K-means
Z = zeros(d, m);
t = 0;
for c = 1:C
    [centroid, ~] = vl_kmeans(X(:,y==c), nc, 'Initialization', 'plusplus', 'NumRepetitions', 3);
    Z(:, t+1:t+nc) = centroid;
    t = t + nc;
end
clear centroid;

%initialize W
if(0)
    W = ITQtrain(X', K); %unsupervised initialization
else
    W = ITQCCAtrain(X', y', K); %supervised initialization
end

%initialize theta
if(isfield(opts,'theta'))
    theta = opts.theta;
else
    B = compactbit(X'*W>0)';
    D2 = hammDist_mex(B, B);
    theta = 1/(opts.theta_scl*mean(D2(:)));
    clear kX B D2;
end

if(isfield(opts,'maxiter'))
    maxiter = opts.maxiter;
else
    maxiter = 300;
end

%optimization
out = call_lbfgs([W(:);Z(:)]', X, R, opts.appr_type, alpha, theta, maxiter);

W = reshape(out(1:d*K), d, K);
Z = reshape(out(d*K+1:end), d, m);

model.tim = toc;
model.W = W;
model.Z = Z;
model.yhat = yhat;
model.theta = theta;
model.normlize_info = normlize_info;

end

function out = call_lbfgs(in, X, R, appr_type, alpha, theta, maxiter)

f = @(x) fun_lin(x, X, R, appr_type, alpha, theta);
opts.x0 = in(:);
opts.maxIts = maxiter;
opts.printEvery = 50;
out = lbfgsb(f, -inf(numel(in),1), inf(numel(in),1), opts);

end
