function B = SDHtest(X, model, type)
%X: each row is a sample
%B: each column is a (compact) binary code

if(nargin<3)
    type = 'uint8';
end

X = L2_normalize(X);

Phi_testdata = exp(-Euclid2(X,model.anchor,'row',0)/(2*model.sigma*model.sigma));
Phi_testdata = [Phi_testdata, ones(size(Phi_testdata,1),1)];
if(strcmp(type,'binary'))
    B = (Phi_testdata*model.W>0)';
elseif(strcmp(type,'uint8'))
    B = compactbit(Phi_testdata*model.W>0)';
else
    error('Bad Type!');
end

end