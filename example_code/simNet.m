%   simNet.m
%
%   Syntax:
%       y = simNet(net,x,option)
%
%   Description:
%       This m-file simulates the output of an single hidden layer
%       neural network. The neural network can be of standard feedforward
%       type or of radial basis function type. When a third argument is
%       supplied, the outputs of the hidden neurons will be given as output
%       as well.
%
%   Construction date:  01-08-2006
%   Last updated:       01-08-2006
%
%   E. de Weerdt
%   TUDelft, Faculty of Aerospace Engineering, ASTI & Control and
%   Simulation Division
%
%   Work part of microned-MISAT project
function  output = simNet(net,x,option) %#ok<INUSD>

%   Checking input format
if size(x,1) ~= size(net.range,1)
    disp('Input incorrectly sized...')
    return
end

if strcmp(net.name{1,1},'feedforward')
    %   Generating input for the hidden layer
    n   = size(x,2);
    V1  = [net.IW,net.b{1,1}]*[x;ones(1,n)];

    %   Generating output of the hidden layer
    Y1  = feval(net.trainFunct{1,1},V1);

    %   Generating input for the output layer
    V2  = [net.LW,net.b{2,1}]*[Y1;ones(1,n)];

    %   Generating output of the hidden layer
    Y2  = feval(net.trainFunct{2,1},V2);

elseif strcmp(net.name{1,1},'rbf')
    %   Generating input for the hidden layer
    Nin     = size(x,1);
    L_end   = size(x,2);
    Nhidden = size(net.centers,1);
    V1      = zeros(Nhidden,L_end);
    for i = 1:Nin
        V1 = V1 + (net.IW(:,i)*x(i,:)-(net.IW(:,i).*net.centers(:,i))*ones(1,L_end)).^2;
    end 

    %   Generating output of the hidden layer
    Y1  = exp(-V1);

    %   Generating output for the output layer
    Y2  = net.LW*Y1;

else
    disp('<simNet.m> Supplied network type is not correct. Must be a feedforward or rbf network ...')
    return
end

%   Constructing output structure
if nargin == 2
    output    = Y2;
elseif nargin == 3
    output.Y2 = Y2;
    output.Y1 = Y1;
    output.V1 = V1;
end

%--------------------------- end of file ----------------------------------
