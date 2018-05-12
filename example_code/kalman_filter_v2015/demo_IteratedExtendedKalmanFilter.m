%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of the Linear Kalman Filter
%
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;


% Use these commands to initialize the randomizer to a fixed (reproducable) state.
% rng('default'); % init randomizer (default, fixed)-> version 2014a,b
% RandStream.setDefaultStream(RandStream('mt19937ar','seed', 300));-> version 2013a,b

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n               = 1;
dt              = 0.01;
N               = 1000;
epsilon         = 1e-10;
doIEKF          = 1;
maxIterations   = 100;


printfigs = 0;
figpath = '';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set initial values for states and statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ex_0    = 10; % initial estimate of optimal value of x_k_1k_1
x_0     = 5; % initial state
n       = 1; % number of states
nm      = 1; % number of measurements
m       = 1; % number of inputs

B       = [1]; % input matrix
G       = [1]; % noise input matrix

% Initial estimate for covariance matrix
stdx_0  = 10;
P_0     = stdx_0^2;

% System noise statistics:
Ew = 0; % bias
stdw = 1; % noise variance
Q = stdw^2;
w_k = stdw * randn(n, N) + Ew;

% Measurement noise statistics:
Ev = 0; % bias
stdv = 5; % noise variance
R = stdv^2;
v_k = stdv * randn(n, N) + Ev;

% for numerical demo only
w_k(1) = -2;
v_k(1) = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate batch with measurement data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

% Real simulated state-variable and measurements data:
x = x_0;
X_k = zeros(n, N);
Z_k = zeros(nm, N);
U_k = zeros(m, N);
for i = 1:N
    dx = kf_calc_f(0, x, U_k(:,i));%subs(x);
    x = x + (dx + w_k(:,i)) * dt;
    
    X_k(:,i) = x; % store state
    Z_k(:,i) = kf_calc_h(0, x, U_k(:,i)) + v_k(:,i); % calculate measurement 
end

XX_k1k1 = zeros(n, N);
PP_k1k1 = zeros(n, N);
STDx_cor = zeros(n, N);
z_pred = zeros(nm, N);
IEKFitcount = zeros(N, 1);

x_k_1k_1 = Ex_0; % x(0|0)=E{x_0}
P_k_1k_1 = P_0; % P(0|0)=P(0)

time1 = toc;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the Extended Kalman filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
% Extended Kalman Filter (EKF)
ti = 0; 
tf = dt;
n = 1;%length(x_k); % n: state dimension

% Run the filter through all N samples
for k = 1:N
    % Prediction x(k+1|k) 
    [t, x_kk_1] = rk4(@kf_calc_f, x_k_1k_1,U_k(:,k), [ti tf]); 

    % z(k+1|k) (predicted output)
    z_kk_1 = kf_calc_h(0, x_kk_1, U_k(:,k)); %x_kk_1.^3; 
    z_pred(:,k) = z_kk_1;

    % Calc Phi(k+1,k) and Gamma(k+1, k)
    Fx = kf_calc_Fx(0, x, U_k(:,k)); % perturbation of f(x,u,t)
    % the continuous to discrete time transformation of Df(x,u,t) and G
    [dummy, Psi] = c2d(Fx, B, dt);   
    [Phi, Gamma] = c2d(Fx, G, dt);   
    
    % P(k+1|k) (prediction covariance matrix)
    P_kk_1 = Phi*P_k_1k_1*Phi' + Gamma*Q*Gamma'; 
    P_pred = diag(P_kk_1);
    stdx_pred = sqrt(diag(P_kk_1));

    
    % Run the Iterated Extended Kalman filter (if doIEKF = 1), else run standard EKF
    if (doIEKF)

        % do the iterative part
        eta2    = x_kk_1;
        err     = 2*epsilon;

        itts    = 0;
        while (err > epsilon)
            if (itts >= maxIterations)
                fprintf('Terminating IEKF: exceeded max iterations (%d)\n', maxIterations);
                break
            end
            itts    = itts + 1;
            eta1    = eta2;

            % Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
            Hx       = kf_calc_Hx(0, eta1, U_k(:,k)); 
            
            % Check observability of state
            if (k == 1 && itts == 1)
                rankHF = kf_calcObsRank(Hx, Fx);
                if (rankHF < n)
                    warning('The current state is not observable; rank of Observability Matrix is %d, should be %d', rankHF, n);
                end
            end
            
            % The innovation matrix
            Ve  = (Hx*P_kk_1*Hx' + R);

            % calculate the Kalman gain matrix
            K       = P_kk_1 * Hx' / Ve;
            % new observation state
            z_p     = kf_calc_h(0, eta1, U_k(:,k)) ;%fpr_calcYm(eta1, u);

            eta2    = x_kk_1 + K * (Z_k(:,k) - z_p - Hx*(x_kk_1 - eta1));
            err     = norm((eta2 - eta1), inf) / norm(eta1, inf);
        end

        IEKFitcount(k)    = itts;
        x_k_1k_1          = eta2;

    else
        % Correction
        Hx = kf_calc_Hx(0, x_kk_1, U_k(:,k)); % perturbation of h(x,u,t)
        % Pz(k+1|k) (covariance matrix of innovation)
        Ve = (Hx*P_kk_1 * Hx' + R); 

        % K(k+1) (gain)
        K = P_kk_1 * Hx' / Ve;
        % Calculate optimal state x(k+1|k+1) 
        x_k_1k_1 = x_kk_1 + K * (Z_k(:,k) - z_kk_1); 

    end    
    
    P_k_1k_1 = (eye(n) - K*Hx) * P_kk_1 * (eye(n) - K*Hx)' + K*R*K';  
    P_cor = diag(P_k_1k_1);
    stdx_cor = sqrt(diag(P_k_1k_1));

    % Next step
    ti = tf; 
    tf = tf + dt;
    
    % store results
    XX_k1k1(:,k) = x_k_1k_1;
%     PP_k1k1(k,:) = P_k_1k_1;
    STDx_cor(:,k) = stdx_cor;
end

time2 = toc;

% calculate state estimation error (in real life this is unknown!)
EstErr = XX_k1k1-X_k;

fprintf('IEKF state estimation error RMS = %d, completed run with %d samples in %2.2f seconds.\n', sqrt(mse(EstErr)), N, time2);


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

plotID = 1000;
figure(plotID);
set(plotID, 'Position', [1 550 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(X_k, 'b');
plot(Z_k, 'k');
title('True state (blue) and Measured state (black)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStateMeasurement');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 1001;
figure(plotID);
set(plotID, 'Position', [1 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(X_k, 'b');
plot(XX_k1k1, 'r');
%plot(z_pred, 'r');
title('True state (blue) and Estimated state (red)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStateEstimates');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 1002;
figure(plotID);
set(plotID, 'Position', [500 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(X_k, 'b');
plot(XX_k1k1, 'r');
plot(Z_k, 'k');
title('True state (blue), Estimated state (red), Measured state (black)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 2001;
figure(plotID);
set(plotID, 'Position', [800 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(EstErr, 'b');
title('State estimation error');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 2002;
figure(plotID);
set(plotID, 'Position', [800 550 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(EstErr, 'b');
axis([0 50 min(EstErr) max(EstErr)]);
title('State estimation error (Zoomed in)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 2003;
figure(plotID);
set(plotID, 'Position', [1000 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(EstErr, 'b');
plot(STDx_cor, 'r');
plot(-STDx_cor, 'g');
legend('Estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
title('State estimation error with STD of Innovation');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 2004;
figure(plotID);
set(plotID, 'Position', [1000 550 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(EstErr, 'b');
plot(STDx_cor, 'r');
plot(-STDx_cor, 'g');
axis([0 50 min(EstErr) max(EstErr)]);
if (doIEKF == 1)
    title('Iterated EKF State estimation error');
else
    title('EKF State estimation error');
end
legend('Estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
if (printfigs == 2004)
    if (doIEKF == 1)
        fpath = sprintf('fig_demoEKFStatesEstimatesError');
    else
        fpath = sprintf('fig_demoIEKFStatesEstimatesError');
    end
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 3001;
figure(plotID);
set(plotID, 'Position', [1 700 600 300], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(IEKFitcount, 'b');
title('IEKF iterations at each sample');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end
