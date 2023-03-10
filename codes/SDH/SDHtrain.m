function model = SDHtrain(X, y, nbits, n_anchors, sigma_scl)
%X: each row is a sample
%y: column vector

tic;
X = L2_normalize(X); %normalization

[N,d] = size(X);
anchor = X(randsample(N, n_anchors),:);
Dis = Euclid2(X, anchor, 'row', 0);
sigma = mean(min(Dis,[],2).^0.5)*sigma_scl;
PhiX = exp(-Dis/(2*sigma*sigma));
PhiX = [PhiX, ones(N,1)];

maxItr = 5;
gmap.lambda = 1; gmap.loss = 'L2';
Fmap.type = 'RBF';
Fmap.nu = 1e-5; %penalty parm for F term
Fmap.lambda = 1e-2;

% Init Z
randn('seed',3);
Zinit = sign(randn(N,nbits));
[~, model, ~] = SDH(PhiX,y,Zinit,gmap,Fmap,[],maxItr,0);
model.tim = toc;
model.anchor = anchor;
model.sigma = sigma;

end