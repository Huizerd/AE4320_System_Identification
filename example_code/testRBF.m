% generate data
[X,Xtrain,Ytrain,fig] = data_generator();
%---------------------------------
% choose a spread constant
spread = .2;
% choose max number of neurons
K = 40;
% performance goal (SSE)
goal = 0;
% number of neurons to add between displays
Ki = 5;
% create a neural network
net = newrb(Xtrain,Ytrain,goal,spread,K,Ki);
%---------------------------------
% view net
view (net)
% simulate a network over complete input range
Y = net(X);
% plot network response
figure(fig)
plot(X,Y,'r')
legend('original function','available data','RBFN','location','northwest')