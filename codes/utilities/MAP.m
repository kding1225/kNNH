function [map, apall] = MAP(traingnd, testgnd, IX)
% mean average precision (MAP) calculation
%
%INPUT:
% traingnd, testgnd: each column is a label vector
% IX: ranked list, ntrain*ntest matrix
%
%OUTPUT:
% map: mean average precision
% apall: average precision of all test samples

[numtrain, numtest] = size(IX);

apall = zeros(1,numtest);
for i = 1 : numtest
    y = IX(:,i);
    x=0;
    p=0;
    new_label=zeros(1,numtrain);
    new_label(testgnd(:,i)'*traingnd>1e-5)=1;
    
    num_return_NN = numtrain;
    for j=1:num_return_NN
        if new_label(y(j))==1
            x=x+1;
            p=p+x/j;
        end
    end  
    if p==0
        apall(i)=0;
    else
        apall(i)=p/x;
    end
end
map = mean(apall);
end
