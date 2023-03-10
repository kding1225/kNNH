function model = k2NNH(X, y, opts)
%train the kernel kNN hashing model, Gaussian kernel is adopted
%
%INPUT:
%X: data matrix, each column is a data point
%y: labels, row vector with each entry in {1,2,...,C}, C is the number of
%   classes
%opts:
% -K: number of hash bits
% -nc: number of prototypes learned for each class
% -appr_type: type of approximation function for sign function, see
%             fun_ker.m for more details
% -gamma_scl: scaling factor of gamma
% -theta: parameter controlling the number of neighbors in defining \Pi_{ij}
% -theta_scl: scaling factor of theta, it should be given if theta is not
%             provided
% -alpha: trade-off parameter
%
%OUTPUT:
%model:
% -A: projection matrix, each column corresponds a hash function
% -Z: prototypes, each column is a prototype
% -yhat: labels of prototypes, row vector
% -gamma: the inverse kernel width
% -theta: parameter controlling the number of neighbors in defining \pi_{ij}
% -normlize_info: information used for normalizing the training data
% -tim: training time

%The code is implemented by Kun Ding (kding@nlpr.ia.ac.cn).

tic;

%data pre-processing
[X, normlize_info] = data_normalization(X, opts.norm_type);

[d, n] = size(X);
C = length(unique(y)); %number of classes
nc = opts.nc; %number of prototypes for each class
K = opts.K; %K-bit hash
m = nc*C; %total number of prototypes
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

%compute gamma
D2 = Euclid2(X, Z, 'col', 0);
gamma = 1/(opts.gamma_scl*sqrt(mean(D2(:)))); %inver kernel width
clear D2;

%initialize A
if(0)
    W = ITQtrain(X', K); %unsupervised initialization
else
    W = ITQCCAtrain(X', y', K); %supervised initialization
end
A = Z'*W;
clear W;

%initialize theta
if(isfield(opts,'theta'))
    theta = opts.theta;
else
    D2 = Euclid2(Z, X, 'col', 0);
    kX = exp(-0.5*gamma^2*D2);
    B = compactbit(kX'*A>0)';
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
out = call_lbfgs([A(:);Z(:)]', X, R, gamma^2, opts.appr_type, alpha, theta^2, maxiter);
A = reshape(out(1:m*K), m, K);
Z = reshape(out(m*K+1:end), d, m);

model.tim = toc;
model.A = A;
model.Z = Z;
model.yhat = yhat;
model.theta = theta;
model.gamma = gamma;
model.normlize_info = normlize_info;

end

function out = call_lbfgs(in, X, R, gamma_2, appr_type, alpha, theta_2, maxiter)

f = @(x) fun_ker(x, X, R, gamma_2, appr_type, alpha, theta_2);
opts.x0 = in(:);
opts.maxIts = maxiter;
opts.printEvery = 50;
out = lbfgsb(f, -inf(numel(in),1), inf(numel(in),1), opts);

end

