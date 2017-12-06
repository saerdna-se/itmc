function [ model ] = gibbs_param( Phi, Psi, Sigma, V, Lambda, l, T )
%GIBBS_PARAM
% Andreas Svensson, 2016

    nb = size(V,1);
    nx = size(Phi,1);
    
    M = zeros(nx,nb);

    Phibar = Phi + (M/V)*M';
    Psibar = Psi +  M/V;
    Sigbar = Sigma + inv(V);
    cov_M = Lambda+Phibar-(Psibar/Sigbar)*Psibar';
    cov_M_sym = 0.5*(cov_M + cov_M'); % To ensure symmetric
    Q = iwishrnd(cov_M_sym,T+l);
    X = randn(nx,nb);
    post_mean = Psibar/Sigbar;
    A = post_mean + chol(Q)*X*chol(inv(Sigbar));

    model.A = A;
    model.Q = Q;

end

