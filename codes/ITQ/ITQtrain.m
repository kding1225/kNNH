function W = ITQtrain(X, nbit)
%X: each row of X is a zero-centered sample
%nbit: number of hashing bit

% PCA, unsupervised
[W, ~] = eigs(cov(X),nbit);
X = X * W;

%ITQ
[~, R] = ITQ(X, 50);%compute ratation matrix

%total transformation matrix
W = real(W*R);

end
