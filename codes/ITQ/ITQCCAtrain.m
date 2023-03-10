function W = ITQCCAtrain(X, y, nbit)
%X: each row of X is a zero-centered sample
%y: labels
%nbit: number of hashing bits

L = unique(y);
Y = sparse(y,1:length(y),ones(1,length(y)),length(L),length(y))';

%CCA, supervised
[W,r] = cca(X, Y, 1e-6); % this computes CCA projections
W = W(:,1:nbit)*diag(r(1:nbit)); % this performs a scaling using eigenvalues
X = X*W; % final projection to obtain embedding E

%ITQ
[~, R] = ITQ(X, 50);%compute ratation matrix

%total transformation matrix
W = real(W*R);

end
