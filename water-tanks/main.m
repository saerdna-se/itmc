% Source code for the real data ater tank example in
% "Is My Model Flexible Enough? Information-Theoretic Model Check",
% Andreas Svensson, Dave Zachariah, Thomas B. Schön arXiv:1712.02675
%
% Code Andreas Svensson 2017

clear, clf
%% Load data from the file at http://www.nonlinearbenchmark.org/index.html#Tanks
u = uEst;
y = yEst;
T = length(yEst);

rng(1)

Mc = 8;
itmc = zeros(1,Mc);


colors = [120 120 120; 228 3 3; 255 140 0; 255 237 0; 0 128 38; 0 77 255; 117 7 135; 0 0 0]./255;

ya = zeros(T,Mc);

for mc = 1:Mc % mc = 1: do the real data thing, mc > 1 is only synthetic data
    tic
    %% Model
    u = u(1:T);
    y = y(1:T);
    
    f = @(x,u,th) [min(10,x(1,:)) + Ts*(-th(1)*sqrt(min(max(x(1,:),0),10)) - th(5)*min(max(x(1,:),0),10) + th(3)*u);...
                   min(10,x(2,:)) + Ts*( th(1)*sqrt(min(max(x(1,:),0),10)) + th(5)*min(max(x(1,:),0),10) - th(2)*sqrt(min(max(x(2,:),0),10)) - th(6)*min(max(x(2,:),0),10) + th(4)*max(x(1,:)-10,0))];


    g   = @(x,u) min(10,x(2,:));

    nx = 2;

    nlth = 6;
    n_th = nlth+2+1;

    if mc > 1
        %% Generate own data and overwrite imported
        rng(mc)
        thetat = [.04 .042 .036 .06 0 0 .0012 .0012 6]' + (.5-rand(9,1)).*[.007 .007 .007 .007 .003 .003 0.0005 0.0005 .2]';

        x = zeros(2,T+1);
        x(:,1) = [thetat(nlth+3), y(1)]';
        y = zeros(T,1);
        for t = 1:T
            x(:,t+1) = f(x(:,t),u(t),thetat) + mvnrnd([0 0],thetat(nlth+1)*eye(2))';
            y(t) = g(x(:,t),u(t)) + mvnrnd(0,thetat(nlth+2));
        end
    end
    ya(:,mc) = y;
    %%           
    % theta: 1:n_lin_th K, 7:8 Q R, 9 x1(1)

    regularization = 1000*ones(1,nlth);
    Vi = diag(regularization);
	Vi_Y_inv = 0;
    lQ = 0; % Flat priors for variance parameters
    LambdaQ = 0; % Flat priors for variance parameters
    lR = 0; % Flat priors for variance parameters
    LambdaR = 0; % Flat priors for variance parameters

    x2i = y(1);

    %% PMCMC
    tic
    rng(mc)

    K = 1000;
    theta = zeros(n_th,K);
    % theta(:,1) = [.04 .042 .036 .06 zeros(1,2*M) .1 .1]';
    theta(:,1) = [.05+0.02*randn(1,4), 0, 0, .1, .1, 6+randn]';
    if mc>1
        theta(:,1) = thetat;
    end

    x_prim = zeros(nx,1,T);

    N = 100;

    Phi_Y = y(1:end)'*y(1:end);

    init_sa = zeros(1,K);

    for k = 1:K

        Q = eye(nx)*theta(nlth+1,k);
        Q_chol = chol(Q);
        R = theta(8,k);
        f_i = @(x,u) f(x,u,theta(:,k));
        g_i = g;

        % Run CPF
        % Initialize the particles
        x = zeros(nx,N,T);
        a = zeros(T,N);
        w = zeros(T,N);
        x(:,:,1) = repmat([theta(9,k), x2i]',[1 N]) + [randn(1,N)*0.1; zeros(1,N)];
        if k > 1
            x(:,N,:) = x_prim; % Set the conditional trajectory
        end

        for t = 1:T

            % PF time propagation, resampling and ancestor sampling
            if t >= 2
                if k > 1
                    a(t,1:N-1) = systematic_resampling(w(t-1,:),N-1);
                    x(:,1:N-1,t) = f_i(x(:,a(t,1:N-1),t-1),u(t-1)) + Q_chol*randn(nx,N-1);

                    waN = w(t-1,:).*mvnpdf(f_i(x(:,:,t-1),u(t-1))',x(:,N,t)',Q)';
                    waN = waN./sum(waN); a(t,N) = systematic_resampling(waN,1);
                else % Run a standard PF on first iteration
                    a(t,:) = multinomial_resampling(w(t-1,:),N);
                    x(:,:,t) = f_i(x(:,a(t,:),t-1),u(t-1)) + Q_chol*randn(nx,N);
                end
            end

            % PF weight update
            log_w = -(g_i(x(:,:,t),u(t)) - y(t)).^2/2/R; 
            w(t,:) = exp(log_w - max(log_w));
            w(t,:) = w(t,:)/sum(w(t,:));
        end


        % Sample new conditional trajectory

        star = systematic_resampling(w(end,:),1);
        x_prim(:,1,T) = x(:,star,T);
        for t = T-1:-1:1
            star = a(t+1,star);
            x_prim(:,1,t) = x(:,star,t);
        end


        % Compute suff stat

        zeta1 = squeeze(x_prim(1,1,2:T))' - min(10,squeeze(x_prim(1,1,1:T-1)))';
        zeta2 = squeeze(x_prim(2,1,2:T))' - min(10,squeeze(x_prim(2,1,1:T-1)))';

        zeta = [zeta1, zeta2];

        z1 = Ts*[-squeeze(sqrt(min(max(x_prim(1,1,1:T-1),0),10)))';
                 zeros(1,T-1);
                 u(1:T-1)';
                 zeros(1,T-1);
                 -squeeze(min(max(x_prim(1,1,1:T-1),0),10))';
                 zeros(1,T-1)];

        z2 = Ts*[squeeze(sqrt(min(max(x_prim(1,1,1:T-1),0),10)))';
                 -squeeze(sqrt(min(max(x_prim(2,1,1:T-1),0),10)))';
                 zeros(1,T-1);
                 squeeze(max(x_prim(1,1,1:T-1)-10,0))';
                 squeeze(min(max(x_prim(1,1,1:T-1),0),10))';
                 -squeeze(min(max(x_prim(2,1,1:T-1),0),10))'];

        z = [z1,z2];

        Phi = (zeta*zeta');
        Psi = (zeta*z');
        Sigma = (z*z');

        Psi_Y = (y(1:end)'*squeeze(x_prim(2,1,1:end)));
        Sigma_Y = (squeeze(x_prim(2,1,1:end))'*squeeze(x_prim(2,1,1:end)));

        % Sample new parameter

        model = gibbs_param( Phi, Psi, Sigma, Vi, LambdaQ,lQ,2*T-2);

        theta(1:nlth,k+1) = model.A;
        theta(nlth+1,k+1) = model.Q;
        theta(nlth+2,k+1) = iwishrnd(LambdaR + Phi_Y-(Psi_Y/(Sigma_Y+Vi_Y_inv))*Psi_Y,T+lR);

        theta(nlth+3,k+1) = x_prim(1,1,1);

        if round(k/100)*100 == k
            disp(['inference, mc = ',num2str(mc),',k = ', num2str(k)])
        end
    end
%     toc
    disp(['Inference done, mc = ',num2str(mc)])
    %%

    burn_in = 400;

    subplot(211)
    hold off
    plot(theta(1:8,:)')
    hold on
    plot([burn_in burn_in],ylim,'k--')

    drawnow

    T = length(u);

    % Evaluate ITMC
    N = 800;

    x = zeros(nx,N,T);
    a = zeros(T,N);
    w = zeros(T,N);
    log_inc = zeros(1,T);
    
    Nth = 30;
    My = 100;
    
    log_likelihood_sim = zeros(Nth,My);
    log_likelihood_data = zeros(Nth,1);
    two_min_thing = zeros(Nth,1);
    
    for k = 1:Nth
        % Draw a posterior sample
        kb = randi([burn_in K]);
        
        % Evaluate likelihood for original data using a BPF
        Q = eye(nx)*theta(nlth+1,kb);
        Q_chol = chol(Q);
        R = theta(nlth+2,kb);
        x(:,:,1) = repmat([theta(nlth+3,kb) x2i]',[1 N]);
        for t = 1:T
            if t >= 2
                a(t,:) = systematic_resampling(w(t-1,:),N);
                x(:,:,t) = f(x(:,a(t,:),t-1),u(t-1),theta(:,kb)) + Q_chol*randn(nx,N);
            end
            log_w = -log(R)-(g_i(x(:,:,t),u(t)) - y(t)).^2/2/R;
            log_inc(1,t) = log(1/N*sum(exp(log_w)));
            if log_inc(1,t)==-Inf
                disp('!')
            end
            w(t,:) = exp(log_w - max(log_w));
            w(t,:) = w(t,:)/sum(w(t,:));
        end
        log_likelihood_data(k) = sum(log_inc);
        
        for m = 1:My
            % Simulate data for sample kb
            x_sim = zeros(2,T+1); y_sim = zeros(1,T);
            x_sim(:,1) = [theta(nlth+3,kb), y(1)]';
            for t = 1:T
                x_sim(:,t+1) = f(x_sim(:,t),u(t),theta(:,kb)) + mvnrnd([0 0],theta(nlth+1,kb)*eye(2))';
                y_sim(:,t) = g(x_sim(:,t),u(t)) + mvnrnd(0,theta(nlth+2,kb));
            end

            % Evaluate likelihood for simulated data using a BPF
            Q = eye(nx)*theta(nlth+1,kb);
            Q_chol = chol(Q);
            R = theta(nlth+2,kb);
            x(:,:,1) = repmat([theta(nlth+3,kb) x2i]',[1 N]);
            for t = 1:T
                if t >= 2
                    a(t,:) = systematic_resampling(w(t-1,:),N);
                    x(:,:,t) = f(x(:,a(t,:),t-1),u(t-1),theta(:,kb)) + Q_chol*randn(nx,N);
                end
                log_w = -log(R)-(g_i(x(:,:,t),u(t)) - y_sim(t)).^2/2/R;
                log_inc(1,t) = log(1/N*sum(exp(log_w)));
                w(t,:) = exp(log_w - max(log_w));
                w(t,:) = w(t,:)/sum(w(t,:));
            end
            log_likelihood_sim(k,m) = sum(log_inc);
        end
        
        log_likelihood_sim_g_data = mean(log_likelihood_data(k)<log_likelihood_sim(k,:));
        two_min_thing(k) = 2*min([log_likelihood_sim_g_data,1-log_likelihood_sim_g_data]);
        
        if round(k/10)*10 == k
            disp(['evaluation, mc = ',num2str(mc),',k = ', num2str(k)])
        end
    end
    
    itmc(mc) = mean(two_min_thing);
    
    disp(['Metric: ', num2str(itmc(mc))])
    
    subplot(223)
    plot((1:T)*Ts,y,'color',colors(mc,:),'linewidth',2)
    hold on
    
    subplot(224)
    plot(itmc(mc)*[1 1],[0 1],'color',colors(mc,:),'linewidth',2)
    hold on
    
    drawnow
end
csvwrite('itmc.csv',itmc)
csvwrite('y.csv',ya)
