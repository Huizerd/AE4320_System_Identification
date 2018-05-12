
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of least squares estimators
%
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

RandStream.setDefaultStream(RandStream('mt19937ar','seed', 181));

printfigs = 0;
doanimation = 0;
saveanimation = 0;
figpath = 'D:\ccdevisser\UD\Courses\AE4312\Lectures\Parameter Estimation\figures\';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set measurement data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 25; % initial measurement points
Nn = 200;

NVAL = 200; % validation points


noiseStd0 = .1; % standard deviation of noise signal


minx = 0;
maxx = 3;

eminx = 1.5;
emaxx = 3;

totalRealizations = 100;


% random sample locations 
% x = (maxx-minx)*(rand(N, 1))+minx;
% gridded sampling locations
x = (minx:(maxx-minx)/(N-1):maxx)';
xnew = (minx:(maxx-minx)/(Nn-1):maxx)';
% system measurements
Y = 20*exp(-.2*(x-1.5).^2);
Ynew = 20*exp(-.2*(xnew-1.5).^2) - 10*exp(-.1*(xnew+2).^2);



% validation data locations and values
xval = (minx:(maxx-minx)/(NVAL-1):maxx)';
Yval = 20*exp(-.2*(xval-1.5).^2); %-.75*xval.^2 + 5*xval -.2;
Yvalnew = 20*exp(-.2*(xval-1.5).^2) - 10*exp(-.1*(xval+2).^2);


sensornoise = noiseStd0*randn(N, 1);
Y = Y + sensornoise;

sensornoisen = noiseStd0*randn(Nn, 1);
Ynew = Ynew + sensornoisen;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create linear regression model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we assume quadratic polynomial model: p(x,theta) = theta_0 + theta_1*x + theta_2*x^2
% create regression matrix X such that p(x,theta) = X * theta
X = [ones(size(x)) x x.^2 x.^3]; % for identification
Xnew = [ones(size(xnew)) xnew xnew.^2 xnew.^3]; % for identification

Xval = [ones(size(xval)) xval xval.^2 xval.^3]; % for validation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do initial least squares estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we assume quadratic polynomial model: p(x,theta) = theta_0 + theta_1*x + theta_2*x^2

COV = inv(X'*X);
c_OLS = COV * X' * Y;

% estimate least squares model using both old and new data
COVnew = inv([X; Xnew]'*[X; Xnew]);
c_OLSnew = COVnew * [X; Xnew]' * [Y; Ynew];

var_c_OLS = diag(COV);

YidOLS = X * c_OLS;
YvalOLS = Xval * c_OLS;
YvalOLSnew = Xval * c_OLSnew;

errOLS = Y-YidOLS;
errvalOLS = Yval-YvalOLS;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do recursive least squares loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P_kp1 = COV;
c_kp1 = c_OLS;
lambda = 1;

plotID = 101;
figure(plotID);
set(plotID, 'Position', [100 150 600 400], 'defaultaxesfontsize', 12, 'defaulttextfontsize', 12, 'PaperPositionMode', 'auto');
for i = 1:size(xnew, 1)
    x_kp1 = xnew(i);
    y_kp1  = Ynew(i);
    
    P_k = P_kp1;
    c_k = c_kp1;
    
    % new row in regression matrix
    a_kp1 = [ones(size(x_kp1)) x_kp1 x_kp1.^2 x_kp1.^3]; 

    % precalculate the a_kp1 * P_k matrix for speed (6x improvement)
    a_kp1P_k = a_kp1 * P_k;
    % calculate Kalman gain
    K_kp1 = (P_k * a_kp1') / (lambda + a_kp1P_k * a_kp1');
     % update covariance matrix
    P_kupdate = K_kp1 * a_kp1P_k;
    P_kp1 = P_k - P_kupdate;
       
    % update parameters
    delta_c =  K_kp1 * (y_kp1 - a_kp1 * c_k);
    c_kp1 = c_k + delta_c;      
    
%     c_kp1 = c_k + P_k*a_kp1'* (y_kp1 - a_kp1 * c_k) + K_kp1*a_kp1*P_k*a_kp1'*(y_kp1 - a_kp1 * c_k);
    
    c_kp1test4 = P_kp1 * (X'*Y + a_kp1'*y_kp1); %correct
    c_kp1test5 = P_kp1 * (inv(P_k)*c_k + a_kp1'*y_kp1); %correct
    c_kp1test6 = P_kp1 * (inv(P_kp1)*c_k -a_kp1'*a_kp1*c_k+ a_kp1'*y_kp1); %correct
    c_kp1test7 = c_k + P_kp1 * ( -a_kp1'*a_kp1*c_k+ a_kp1'*y_kp1); %correct
    c_kp1test8 = c_k - P_kp1*a_kp1'*(a_kp1*c_k + y_kp1); %correct
    X = [X; a_kp1];
    Y = [Y; y_kp1];

    if (doanimation)
        YRLS = Xnew * c_kp1;
        
        hold off;
        grid on;
        plot(x, Y, 's', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 4);
        hold on;
        plot(xnew(1:i), Ynew(1:i), 'o', 'markerfacecolor', 'r', 'markeredgecolor', 'k', 'markersize', 2);
        plot(xval, YvalOLS, 'b', 'linewidth', 2);
        plot(xval, YRLS, 'r', 'linewidth', 2);
        axis([-.1 3.1 .9*min([YvalOLS; Ynew]) 1.1*max([YvalOLS; Ynew])]);
        xlabel('x');
        ylabel('f(x)');
        title(sprintf('RLS model at iteration %d', i));
%         legend('OLS model', 'RLS model', 'Real System', 'Location', 'southeast');
        drawnow;
        if (saveanimation)
            Movie(i) = getframe(gcf);
            fprintf('Recording frame %d of %i\n', i, length(xnew));
        end
        
    end

end

% save animation to disc
if (saveanimation)
    fname = strcat(figpath, sprintf('demoRLS_Nold%d_Nnew%d', N, Nn));
    mov = avifile(fname, 'Quality', 100, 'Compression', 'none', 'FPS', 20);
    for i = 1:length(Movie)-1
        fprintf('Adding frame %d of %i\n', i, length(xnew));
        mov = addframe(mov, Movie(i));
    end
    clear Movie;
    mov = close(mov);
    disp(sprintf('Saved movie in <%s>', fname));
end

c_RLS = c_kp1;

YidRLS = Xnew * c_RLS;
YvalRLS = Xval * c_RLS;

errRLS = Ynew-YidRLS;
errvalRLS = Yvalnew-YvalRLS;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
close all;

plotID = 999;
figure(plotID);
set(plotID, 'Position', [50 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(x, Y, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 4);
% axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('f(x)');
title('Original Measurement Data');
if (printfigs == 1)
    fpath = sprintf('fig_demoNoisyOriginalData');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 1000;
figure(plotID);
set(plotID, 'Position', [50 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
% plot(xval, Yval, 'r', 'linewidth', 2); 
plot(x, Y, 's', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 4);
plot(xnew, Ynew, 'o', 'markerfacecolor', 'r', 'markeredgecolor', 'k', 'markersize', 2);
% axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('f(x)');
title('Original & New Measurement Data');
legend('original data', 'new data', 'Location', 'southeast');
if (printfigs == 1)
    fpath = sprintf('fig_demoNewData');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

%%
plotID = 1001;
figure(plotID);
set(plotID, 'Position', [50 500 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(xval, YvalOLS, 'b', 'linewidth', 2);
plot(x, Y, 's', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 4);
plot(xnew, Ynew, 'o', 'markerfacecolor', 'r', 'markeredgecolor', 'k', 'markersize', 2);
% axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('f(x)');
title('Data and OLS model');
legend('OLS model', 'real system output', 'data', 'Location', 'southwest');
if (printfigs == 1)
    fpath = sprintf('fig_demoEstOLS');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 2001;
figure(plotID);
set(plotID, 'Position', [500 500 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(xval, YvalOLSnew, 'b--', 'linewidth', 2);
plot(xval, YvalOLS, 'b', 'linewidth', 2);
plot(xval, YvalRLS, 'r', 'linewidth', 2);
plot(xval, Yvalnew, 'k', 'linewidth', 2); 
plot(xnew, Ynew, 'o', 'markerfacecolor', 'r', 'markeredgecolor', 'k', 'markersize', 2);
% axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('f(x)');
title('Data, OLS, and RLS model');
legend('OLS model', 'RLS model', 'Real System', 'Location', 'southeast');
if (printfigs == 1)
    fpath = sprintf('fig_demoEstOLSWLSReal');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

return;

%%
plotID = 2002;
figure(plotID);
set(plotID, 'Position', [800 500 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(xval, YvalOLS, 'b', 'linewidth', 2);
plot(xval, YvalWLS, 'r', 'linewidth', 2);
plot(x, Y, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 4);
% axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('f(x)');
title('Data, OLS, and WLS model');
legend('OLS model', 'WLS model', 'data', 'Location', 'southeast');
if (printfigs == 1)
    fpath = sprintf('fig_demoEstOLSWLS');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 2003;
figure(plotID);
set(plotID, 'Position', [800 500 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(xval, Yval, 'k', 'linewidth', 2); 
plot(x, Y, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 4);
plot(xval, YvalOLS, 'b', 'linewidth', 2);
plot(xval, YvalWLS, 'r', 'linewidth', 2);
% axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('f(x)');
title('Data, real system output, OLS, and WLS model');
legend('real system output', 'data', 'OLS model', 'WLS model', 'Location', 'southeast');
if (printfigs == 1)
    fpath = sprintf('fig_demoEstOLSWLSRealData');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

%%
if (totalRealizations == 12)
        
    plotID = 4001;
    figure(plotID);
    set(plotID, 'Position', [500 100 900 500], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
    for i = 1:totalRealizations
        subplot(3,4,i);
        hold on;
        grid on;
        bar(1:size(OLSset{i,1}), OLSset{i,1});%, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 2);
        axis([0 5 -10 20]);
        title(sprintf('Est. Params. noise real %d', i));
    end
    if (printfigs == 1)
        fpath = sprintf('fig_demoParamsOLSSet');
        savefname = strcat(figpath, fpath);
        print(plotID, '-dpng', '-r300', savefname);
    end

    
    plotID = 4002;
    figure(plotID);
    set(plotID, 'Position', [550 100 900 500], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
    for i = 1:totalRealizations
        subplot(3,4,i);
        hold on;
        grid on;
        plot(x, OLSset{i,3}, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 2);
        plot(xval, OLSset{i,2}, 'b', 'linewidth', 1);
        axis([min(x) max(x) 10 25]);
        title(sprintf('Noise realization %d', i));
    end
    if (printfigs == 1)
        fpath = sprintf('fig_demoEstOLSSet');
        savefname = strcat(figpath, fpath);
        print(plotID, '-dpng', '-r300', savefname);
    end
   
    
    plotID = 4101;
    figure(plotID);
    set(plotID, 'Position', [500 100 900 500], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
    for i = 1:totalRealizations
        subplot(3,4,i);
        hold on;
        grid on;
        bar(1:size(WLSset{i,1}), WLSset{i,1});%, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 2);
        axis([0 5 -10 20]);
        title(sprintf('Est. Params. noise real %d', i));
    end
    if (printfigs == 1)
        fpath = sprintf('fig_demoParamsWLSSet');
        savefname = strcat(figpath, fpath);
        print(plotID, '-dpng', '-r300', savefname);
    end
    
    plotID = 4102;
    figure(plotID);
    set(plotID, 'Position', [700 100 900 500], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
    for i = 1:totalRealizations
        subplot(3,4,i);
        hold on;
        grid on;
        plot(x, WLSset{i,3}, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 2);
        plot(xval, WLSset{i,2}, 'b', 'linewidth', 1);
        axis([min(x) max(x) 10 25]);
        title(sprintf('Noise realization %d', i));
    end
    if (printfigs == 1)
        fpath = sprintf('fig_demoEstWLSSet');
        savefname = strcat(figpath, fpath);
        print(plotID, '-dpng', '-r300', savefname);
    end
    
    %%
    plotID = 4201;
    figure(plotID);
    set(plotID, 'Position', [800 100 900 500], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
    for i = 1:totalRealizations
        subplot(3,4,i);
        hold on;
        grid on;
        bar([(1:size(WLSset{i,1}))' (1:size(WLSset{i,1}))'], [OLSset{i,1} WLSset{i,1}]);%, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 2);
        axis([0 5 -10 20]);
        if (i == totalRealizations)
            legend('OLS params.', 'WLS params.', 'location', [.84 .82 .15 .1]);
        end
        title(sprintf('Est. Params. noise real %d', i));
    end
    if (printfigs == 1)
        fpath = sprintf('fig_demoParamsOLSWLSSet');
        savefname = strcat(figpath, fpath);
        print(plotID, '-dpng', '-r300', savefname);
    end

    
end

