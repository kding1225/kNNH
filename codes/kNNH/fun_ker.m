function [f, g] = fun_ker(in, X, R, gamma_2, appr_type, alpha, theta_2)
%compute function value and gradients needed for L-BFGS based
%optimization, kernel kNN hashing
%
%INPUT:
%in: the variables to be optimized, here, it is the concatenation of A and Z
%X: data matrix, n_dims*n_samples
%R: n_samples*n_anchors matrix
%gamma_2: gamma^2
%appr_type: a cell structure, the first entry is one of 'tanh', 'sqrt' and
%           'lin', which denotes the tanh approximation, the sqrt
%           approximation and the linear approximation, respectively. Note
%           that only the separable approximations are implemented. The
%           second entry (a real number >0) is only used for the 'sqrt' option.
%alpha: the trade-off parameter
%theta_2: theta^2
%
%OUTPUT:
%f: function value
%g: gradients


EP = 1e-6; %small constant

%get inputs
[d, n] = size(X);
m = size(R, 2);
K = numel(in)/m - d; %'in' has m*d+m*K entries
A = reshape(in(1:m*K), m, K);
Z = reshape(in(m*K+1:end), d, m);
clear in;

%compute kernel matrix: k(X) and k(Z)
D2 = Euclid2(Z, X, 'col', 0);
kX = exp(-0.5*gamma_2*D2);
D2 = Euclid2(Z, Z, 'col', 0);
kZ = exp(-0.5*gamma_2*D2);

%compute H(X), H(Z), H'(X), H'(Z)
if(strcmp(appr_type{1},'tanh'))
    hX = tanh(A'*kX);
    hZ = tanh(A'*kZ);
    hX1 = 1 - hX.^2;
    hZ1 = 1 - hZ.^2;
elseif(strcmp(appr_type{1},'sqrt'))
    if(length(appr_type)>1)
        ep = appr_type{2};
    else
        ep = 0.5;
    end
    tmp1 = A'*kX;
    hX = tmp1./sqrt(tmp1.^2+ep);
    tmp2 = A'*kZ;
    hZ = tmp2./sqrt(tmp2.^2+ep);
    hX1 = ep./(tmp1.^2+ep).^(3/2);
    hZ1 = ep./(tmp2.^2+ep).^(3/2);
elseif(strcmp(appr_type{1},'lin'))
    hX = A'*kX;
    hZ = A'*kZ;
    hX1 = ones(size(hX));
    hZ1 = ones(size(hZ));
end
clear tmp1 tmp2;

%compute P, Q, \tilder{R}, \hat{R}, and U
V = hX'*hZ;
P = theta_2*V;
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

%compute objective value
f = -sum(log(p+EP)) - sum(log(q+EP))*alpha; %minimize this value

%compute derivatives
if(nargout>1)
    
    %derivative of A
    Bx = theta_2*(hX*U).*hZ1;
    Bz = theta_2*(hZ*U').*hX1;
    g1 = kX*Bz' + kZ*Bx';
    
    %derivative of Z
    Dx = (A*Bx).*kZ;
    Dz = (A*Bz).*kX;
    g2 = gamma_2*(X*Dz'+Z*(Dx+Dx'-diag(sum(Dx,2)+sum(Dx)'+sum(Dz,2))));
    
    g = [g1(:);g2(:)];
end

end
