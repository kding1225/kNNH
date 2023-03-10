function D2 = Euclid2(A, B, row_col, bSqrt)

if(nargin<3)
    row_col = 'col';
end
if(nargin<4)
    bSqrt = 0; %default, compute squard Euclidean distance
end

if(strcmp(row_col,'col'))
    D2 = sqdist(A, B);
else %row
    D2 = sqdist(A', B');
end
if(bSqrt)
    D2 = sqrt(D2);
end

end

function d = sqdist(a, b)
% SQDIST - computes squared Euclidean distance matrix
%          computes a rectangular matrix of pairwise distances
% between points in A (given in columns) and points in B

% NB: very fast implementation taken from Roland Bunschoten

aa = sum(a.^2,1); bb = sum(b.^2,1); ab = a'*b; 
d = (repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

end
