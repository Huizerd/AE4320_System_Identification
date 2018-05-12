%% Data generator function
function [X,Xtrain,Ytrain,fig] = data_generator()
% data generator
X = 0.01:.01:10;
f = abs(besselj(2,X*7).*asind(X/2) + (X.^1.95)) + 2;
fig = figure;
plot(X,f,'b-')
hold on
grid on
% available data points
Ytrain = f + 5*(rand(1,length(f))-.5);
Xtrain = X([181:450 601:830]);
Ytrain = Ytrain([181:450 601:830]);
plot(Xtrain,Ytrain,'kx')
xlabel('x')
ylabel('y')
ylim([0 100])
legend('original function','available data','location','northwest')