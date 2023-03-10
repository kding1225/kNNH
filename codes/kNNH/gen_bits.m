function B = gen_bits(X, model, kernel_type, bit_type)
%binary encoding of kNNH and k2NNH
%
%INPUT:
%X: each column is a sample
%model: a structure of kNNH or k2NNH model
%kernel_type: 'lin' or 'kernel'
%bit_type: compute 0-1 binary code ('binary') or compact binary code ('uint8', default)
%
%OUTPUT:
%B: each column is an encoding vector

%pre-processing
X = data_normalization(X, model.normlize_info.type, model.normlize_info);

if(nargin<4)
    bit_type = 'uint8';
end

if(strcmp(kernel_type,'linear')) %linear
    if(strcmp(bit_type,'binary'))
        B = model.W'*X>0;
    else %uint8
        B = compactbit(X'*model.W>0)';
    end
elseif(strcmp(kernel_type,'kernel')) %kernel
    D2 = Euclid2(model.Z, X, 'col', 0);
    kX = exp(-0.5*model.gamma^2*D2);
    if(strcmp(bit_type,'binary'))
        B = model.A'*kX>0;
    else %uint8
        B = compactbit(kX'*model.A>0)';
    end
else
    error('Bad type!');
end

end
