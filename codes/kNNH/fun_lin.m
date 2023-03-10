function [f, g] = fun_lin(in, X, R, appr_type, alpha, theta)
%compute function value and gradients needed for L-BFGS based optimization, linear kNN
%hashing
%
%INPUT:
%in: the variables to be optimized, here, it is the concatenation of W and Z
%X: data matrix, n_dims*n_samples
%R: n_samples*n_anchors matrix
%appr_type: a cell structure, the first entry is one of 'tanh', 'sqrt' and
%           'lin', which denotes the tanh approximation, the sqrt
%           approximation and the linear approximation, respectively. Note
%           that only the separable approximations are implemented. The
%           second entry (a real number >0) is only used for the 'sqrt' option.
%alpha: trade-off parameter
%theta_2: theta^2
%
%OUTPUT:
%f: function value
%g: gradients


EP = 1e-6; %small constant

%get inputs
[d, n] = size(X);
m = size(R, 2); %number of anchors
K = numel(in)/d - m;
W = reshape(in(1:d*K), d, K);
Z = reshape(in(d*K+1:end), d, m);
clear in;

%compute H(X), H(Z), H'(X), H'(Z)
if(strcmp(appr_type{1},'tanh'))
    hX = tanh(W'*X);
    hZ = tanh(W'*Z);
    hX1 = 1 - hX.^2;
    hZ1 = 1 - hZ.^2;
elseif(strcmp(appr_type{1},'sqrt'))
    if(length(appr_type)>1)
        ep = appr_type{2};
    else
        ep = 0.5;
    end
    tmp1 = W'*X;
    hX = tmp1./sqrt(tmp1.^2+ep);
    tmp2 = W'*Z;
    hZ = tmp2./sqrt(tmp2.^2+ep);
    hX1 = ep./(tmp1.^2+ep).^(3/2);
    hZ1 = ep./(tmp2.^2+ep).^(3/2);
elseif(strcmp(appr_type{1},'lin'))
    hX = W'*X;
    hZ = W'*Z;
    hX1 = ones(size(hX));
    hZ1 = ones(size(hZ));
end
clear tmp1 tmp2;

%compute P, Q, \tilder{R}, \hat{R}, and U
V = hX'*hZ;
P = theta^2*V;
Q = P;
P = bsxfun(@minus, P, max(P,[],2)); %omit this may incur NaN value
Q = bsxfun(@minus, Q, max(Q,[],1));
P = exp(P);
Q = exp(Q);
P = bsxfun(@rdivide, P, sum(P,2)+EP);
Q = bsxfun(@rdivide, Q, sum(Q,1)+EP);
p = sum(R.*P, 2);
q = sum(R.*Q, 1);
Rtid = bsxfun(@rdivide, R, p+EP);
Rhat = bsxfun(@rdivide, R, q+EP);
U = (1-Rtid).*P + (1-Rhat).*Q*alpha;
clear P Q Rtid Rhat;

%compute objective
f = -sum(log(p+EP)) - sum(log(q+EP))*alpha;

%compute derivatives
if(nargout>1)
    
    %derivative of W
    Bx = theta^2*(hX*U).*hZ1;
    Bz = theta^2*(hZ*U').*hX1;
    g1 = X*Bz' + Z*Bx';
    
    %derivative of Z
    g2 = W*Bx;
    
    g = [g1(:);g2(:)];
end

end
