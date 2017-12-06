function idx = multinomial_resampling(W,N)

W = W/sum(W);
u = sort(rand(N,1));

idx = zeros(N,1);
q = 0;
n = 0;
for i = 1:N
    while q < u(i)
        n = n+1;
        q = q + W(n);
    end
    idx(i) = n;
end



