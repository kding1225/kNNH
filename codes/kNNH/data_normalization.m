function [X, normlize_info] = data_normalization(X, type, normlize_info)
%data pre-processing
%
%INPUT:
%X: each column is a datum
%type: normlization method
% 'max_norm': remove the mean value and normalize the maximal l2-norm
%   of samples to be 1.0
% 'avg_norm': remove the mean value and normalize the mean l2-norm of
%   samples to be 1.0
% 'l2': remove the mean value and normalize every sample to have unit l2
%   norm
% '0-1': normalize the range of every dimension of data to be [0,1]
% 'normal': normalize every dimension to obey the normal Gaussian distribution
%normlize_info:
% if this input is given, X will be normalized by this infomation;
% otherwise, the information used for normalization will be computed first,
% and the input X will be normalized by this information
%
%OUTPUT:
%X: normalized data
%normlize_info:
% -type: normlization method
% when type=='max_norm'
%    -mean_vec: mean value
%    -max_l2norm: maximal l2-norm of samples
% when type=='avg_norm'
%    -mean_vec: 
%    -mean_l2norm: mean l2-norm of samples
% when type=='l2'
%    -mean_vec: 
% when type=='0-1'
%    -min_dims: minimal value of each dimension
%    -max_dims: maximal value of each dimension
% when type=='normal'
%    -mean_vec: 
%    -std_vec: standard deviation of each dimension
% when type=='none'
%    -


if(exist('normlize_info','var')) %if normlize_info is given
    switch type
        case 'max_norm'
            X = bsxfun(@minus, X, normlize_info.mean_vec);
            X = X/normlize_info.max_l2norm;
        case 'avg_norm'
            X = bsxfun(@minus, X, normlize_info.mean_vec);
            X = X/normlize_info.mean_l2norm;
        case 'l2'
            X = bsxfun(@minus, X, normlize_info.mean_vec);
            X = bsxfun(@rdivide, X, sqrt(sum(X.^2))+eps);
        case '0-1'
            range = normlize_info.max_dims - normlize_info.min_dims + eps;
            X = bsxfun(@minus, X, normlize_info.min_dims);
            X = bsxfun(@rdivide, X, range);
        case 'normal'
            X = bsxfun(@minus, X, normlize_info.mean_vec);
            X = bsxfun(@rdivide, X, normlize_info.std_vec+eps);
        case 'none'
            
        otherwise
            error('Bad Type!');
    end
else %not given
    switch type
        case 'max_norm'
            mean_vec = mean(X, 2);
            X = bsxfun(@minus, X, mean_vec);
            max_l2norm = max(sqrt(sum(X.^2))) + eps;
            X = X/max_l2norm;
            normlize_info.mean_vec = mean_vec;
            normlize_info.max_l2norm = max_l2norm;
        case 'avg_norm'
            mean_vec = mean(X, 2);
            X = bsxfun(@minus, X, mean_vec);
            mean_l2norm = mean(sqrt(sum(X.^2))) + eps;
            X = X/mean_l2norm;
            normlize_info.mean_vec = mean_vec;
            normlize_info.mean_l2norm = mean_l2norm;
        case 'l2'
            mean_vec = mean(X, 2);
            X = bsxfun(@minus, X, mean_vec);
            X = bsxfun(@rdivide, X, sqrt(sum(X.^2))+eps);
            normlize_info.mean_vec = mean_vec;
        case '0-1'
            min_dims = min(X, [], 2);
            max_dims = max(X, [], 2);
            range = max_dims - min_dims + eps;
            X = bsxfun(@minus, X, min_dims);
            X = bsxfun(@rdivide, X, range);
            normlize_info.min_dims = min_dims;
            normlize_info.max_dims = max_dims;
        case 'normal'
            mean_vec = mean(X, 2);
            X = bsxfun(@minus, X, mean_vec);
            std_vec = std(X, 0, 2) + eps;
            X = bsxfun(@rdivide, X, std_vec);
            normlize_info.mean_vec = mean_vec;
            normlize_info.std_vec = std_vec;
        case 'none'
            normlize_info = [];
        otherwise
            error('Bad Type!');
    end
    normlize_info.type = type;
end

end