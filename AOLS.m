function [AOLS_index,AOLS_xhat,AOLS_time,AOLS_it,AOLS_res] = AOLS(A,y,k,L,thr)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate a sparse signal x from linear measurements(Real-valued) 
%                          y = A*x
% using Accelerated Orthogonal Least-Squares(AOLS)
%
% Input parameters
% y	    : Measurement vector 
% A	    : Coefficient matrix 
% k	    : Sparsity level
% L	    : Number of selected indices per iteration
% thr   : Desired accuracy
%
% Output parameterss
% AOLS_xhat	    : Estimated signal
% AOLS_index    : Selected support set.
% AOLS_it       : Number of iterations
% AOLS_time     : CPU time
% AOLS_res      : Final residual norm
%
% Written by Abolfazl Hashemi, August 2016
% ECE department, UT Austin, Austin, TX, 78712, US 
% Email: abolfazl@utexas.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[n,m] = size(A);
AOLS_xhat = zeros(m,1);
r = y;
it = 0;
L = min(L,floor(n/k));
max_itr = min(k,floor(n/L));
AOLS_index = zeros(1,max_itr*L);
U = zeros(n,max_itr*L);
T = A;
while norm(r) > thr && it < k
    it = it + 1;
    temp_in=(it - 1)*L;
    if it > 1
        T = T - U(:,temp_in)*(U(:,temp_in)'*A);
    end
    q = (r'*A)./sum(A.*T,1);
    TT = sum(T.^2,1).*(q.^2);
    TT(AOLS_index(1:temp_in))=0;
    [~,sortingIndices] = sort(TT,'descend');
    AOLS_index(temp_in + 1:temp_in+L) = sortingIndices(1:L);
    for i=1:L  
        TEMP = T(:,sortingIndices(i))*q(sortingIndices(i));
        U(:,temp_in + i) = normc(TEMP);
        r = r - TEMP;
        if i==L
            break
        end
        T = T - U(:,temp_in + i)*(U(:,temp_in + i)'*A);
        q = (r' * A) ./ sum(A .* T,1);
    end
end
AOLS_index = reshape(nonzeros(AOLS_index),1,[]);
AOLS_res = norm(r);
AOLS_xhat(AOLS_index) = A(:,AOLS_index) \ y;
AOLS_it = it;
if L > 1
    [~,sortingIndices] = sort(abs(AOLS_xhat),'descend');
    AOLS_index = reshape(sortingIndices(1:k),1,[]);
    AOLS_xhat(AOLS_index) = A(:,AOLS_index) \ y;
    AOLS_res = norm(y - A(:,AOLS_index)*AOLS_xhat(AOLS_index));
end
AOLS_time = toc;
end
