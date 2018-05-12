
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
figpath = 'D:\ccdevisser\UD\Courses\AE4312\Lectures\Parameter Estimation\figures\';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set measurement data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 50; % measurement points
NVAL = 200; % validation points


noiseStd0 = .1; % standard deviation of noise signal
noiseStdE = 5; % standard deviation of noise signal at high noise measurement locaitons


minx = 0;
maxx = 3;

eminx = 1.5;
emaxx = 3;

totalRealizations = 100;


% random sample locations 
% x = (maxx-minx)*(rand(N, 1))+minx;
% gridded sampling locations
x = (minx:(maxx-minx)/(N-1):maxx)';
% system measurements
Y = 20*exp(-.2*(x-1.5).^2);

% normal noise locations
Nidx = find(x < eminx);

% add varying amplitude noise    
Eidx = find(x > eminx & x < emaxx);

% validation data locations and values
xval = (minx:(maxx-minx)/(NVAL-1):maxx)';
Yval = 20*exp(-.2*(xval-1.5).^2); %-.75*xval.^2 + 5*xval -.2;

sensornoise = noiseStd0*randn(N, 1);
sensornoise(Eidx) = noiseStdE*randn(length(Eidx), 1);
Y = Y + sensornoise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create linear regression model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we assume quadratic polynomial model: p(x,theta) = theta_0 + theta_1*x + theta_2*x^2
% create regression matrix X such that p(x,theta) = X * theta
X = [ones(size(x)) x x.^2 x.^3]; % for identification

Xval = [ones(size(xval)) xval xval.^2 xval.^3]; % for validation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do ordinary least squares estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we assume quadratic polynomial model: p(x,theta) = theta_0 + theta_1*x + theta_2*x^2

COV = inv(X'*X);
c_OLS = COV * X' * Y;

var_c_OLS = diag(COV);

YidOLS = X * c_OLS;
YvalOLS = Xval * c_OLS;

errOLS = Y-YidOLS;
errvalOLS = Yval-YvalOLS;

OLSset = cell(totalRealizations, 3);
sum_c_OLS = zeros(size(c_OLS));
for i = 1:totalRealizations
    sensornoise = noiseStd0*randn(N, 1);
    sensornoise(Eidx) = noiseStdE*randn(length(Eidx), 1);
    Yi = Y + sensornoise;
    c_OLSi = COV * X' * Yi;
    YvalOLSi = Xval * c_OLSi;
    OLSset{i,1} = c_OLSi;
    OLSset{i,2} = YvalOLSi;
    OLSset{i,3} = Yi;
   
    sum_c_OLS = sum_c_OLS + c_OLSi;
end

% empirical variance calculations
empvar_c_OLS = var([OLSset{1:totalRealizations,1}], 0, 2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do weighted least squares estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W = diag(noiseStd0*ones(N,1));
W(Eidx,Eidx) = noiseStdE*eye(length(Eidx));
    

%W = diag(VarY); % define the weighting matrix
Winv = inv(W);

COVWLS = inv(X'*Winv*X);
c_WLS = COVWLS * X'*Winv * Y;

var_c_WLS = diag(COVWLS);

YidWLS = X * c_WLS;
YvalWLS = Xval * c_WLS;

errWLS = Y-YidWLS;
errvalWLS = Yval-YvalWLS;


WLSset = cell(totalRealizations, 3);
sum_c_WLS = zeros(size(c_WLS));
for i = 1:totalRealizations
    sensornoise = noiseStd0*randn(N, 1);
    sensornoise(Eidx) = noiseStdE*randn(length(Eidx), 1);
    Yi = Y + sensornoise;
    c_WLSi = COVWLS * X'*Winv * Y;
    YvalWLSi = Xval * c_WLSi;
    WLSset{i,1} = c_WLSi;
    WLSset{i,2} = YvalWLSi;
    WLSset{i,3} = Yi;
end

% empirical variance calculations
empvar_c_WLS = var([WLSset{1:totalRealizations,1}], 0, 2);




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
title('Measurement Data');
if (printfigs == 1)
    fpath = sprintf('fig_demoNoisyEstimationData');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 1000;
figure(plotID);
set(plotID, 'Position', [50 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(xval, Yval, 'r', 'linewidth', 2); 
plot(x, Y, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 4);
% axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('f(x)');
title('Measurement Data + Actual System Output');
legend('measurement data', 'real system output', 'Location', 'southeast');
if (printfigs == 1)
    fpath = sprintf('fig_demoNoisyEstimationDataSystem');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 1001;
figure(plotID);
set(plotID, 'Position', [50 500 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(xval, YvalOLS, 'b', 'linewidth', 2);
plot(xval, Yval, 'r', 'linewidth', 2); 
plot(x, Y, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 4);
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


plotID = 1002;
figure(plotID);
set(plotID, 'Position', [50 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(x, errOLS, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 4);
% axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('f(x)');
title('OLS model error');
legend('OLS model', 'data', 'Location', 'southeast');
if (printfigs == 1)
    fpath = sprintf('fig_demoEstErrOLS');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 1003;
figure(plotID);
set(plotID, 'Position', [50 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
% plot(x, errOLS, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 4);
% plot(x, errWLS, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'r', 'markersize', 4);
plot(xval, errvalOLS, 'b', 'linewidth', 2);
plot(xval, errvalWLS, 'r', 'linewidth', 2);
% p1=plot([xval(1) xval(end)], [(StdY(1)) (StdY(1))], 'Color', 'g', 'linewidth', 2);
% p2=plot([xval(1) xval(end)], [-(StdY(1)) -(StdY(1))], 'Color', 'g', 'linewidth', 2);
% axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('\epsilon(x)');
title('OLS and WLS model validation residuals');
legend('OLS residuals', 'WLS residuals', 'Location', 'southeast');
if (printfigs == 1)
    fpath = sprintf('fig_demoEstErrOLSWLS');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 1004;
figure(plotID);
set(plotID, 'Position', [50 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
% plot(x, errOLS, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 4);
% plot(x, errWLS, 'o', 'markerfacecolor', 'k', 'markeredgecolor', 'r', 'markersize', 4);
plot(x(Nidx), errOLS(Nidx), 'b', 'linewidth', 2);
plot(x(Nidx), errWLS(Nidx), 'r', 'linewidth', 2);
% p1=plot([x(1) x(Nidx(end))], [(StdY(1)) (StdY(1))], 'Color', 'g', 'linewidth', 2);
% p2=plot([x(1) x(Nidx(end))], [-(StdY(1)) -(StdY(1))], 'Color', 'g', 'linewidth', 2);
% axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('\epsilon(x)');
title('OLS and WLS model residuals');
legend('OLS residuals', 'WLS residuals', 'Location', 'southeast');
if (printfigs == 1)
    fpath = sprintf('fig_demoEstErrOLSWLSzoom');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 2001;
figure(plotID);
set(plotID, 'Position', [500 500 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(xval, YvalOLS, 'b', 'linewidth', 2);
plot(xval, YvalWLS, 'r', 'linewidth', 2);
plot(xval, Yval, 'k', 'linewidth', 2); 
% axis([-.1 3.1 -1.5 5.5]);
xlabel('x');
ylabel('f(x)');
title('Data, OLS, and WLS model');
legend('OLS model', 'WLS model', 'Real System', 'Location', 'southeast');
if (printfigs == 1)
    fpath = sprintf('fig_demoEstOLSWLSReal');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

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

