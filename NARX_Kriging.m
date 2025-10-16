%% MeanRes attempt 6 (Perfect!)
clear all 
clc

tic 

Data = load('Kriging_Kneading.txt');
x = Data(:,1);
y = Data(:,2);
t = Data(:,3);
z = Data(:,4);   % mean res

range = 78.9627;    % Example range for variogram (Calculated from relevant dataset)
sill = 5357.8;    % Example sill for variogram (Calculated from relevant dataset)
L=10;           % Number of interpolated points
% Perform Kriging interpolation
[elevation, gridX, gridY, gridT] = kriging3D(x, y, t, z, range, sill, L);
% To adjust the number of interpolated point (L), edit the function
InputDelay = 1:1;
feedbackDelay = 1:1;
TrainingAlgorithm = 'trainbr';

% Flatten the outputs into a matrix with 4 columns (x, y, t, z)
[gridXFlat, gridYFlat, gridTFlat] = ndgrid(gridX, gridY, gridT);
interpolatedData = [gridXFlat(:), gridYFlat(:), gridTFlat(:), elevation(:)];

%% Begin the Modeling

Data = interpolatedData;

lw = 0.5;
[entries,attributes] = size(Data);
entries_breakpoint = round(entries*.50); 
Data_inputs = Data(:,[1:3]); 
Data_output = Data(:, 4); 
% testData_inputs = Data(entries_breakpoint:end, [2,4,5,6]);
% testData_output = Data(entries_breakpoint:end, 7);


%List of training functions:
%   trainlm         %   
%   trainbfg        %
%   trainrp         %%%
%   trainscg        %
%   traincgb        %%%
%   traincgf        %%%
%   traincgp        %
%   trainposs       %
%   traingdx        %

% hidden layer size (h)
h = [5,5] ;
% Learning rate a
a = 0.007;
% Create a NAR neural network
net = narxnet(InputDelay, feedbackDelay, h,'open', TrainingAlgorithm);  % Modify the architecture as needed
                                               %narxnet(1:numInputDelays, 1:numFeedbackDelays, hiddenlayersize(50), feedbackMode, trainFunction)
                                               %  See also TRAINGDM, TRAINGDA, TRAINGDX, TRAINLM, TRAINRP,
                                               %                     TRAINCGF, TRAINCGB, TRAINBFG, TRAINCGP, TRAINOSS.
            
% Configure the network
net.layers{1}.transferFcn = 'tansig';  % You can choose other activation functions
net.layers{2}.transferFcn = 'tansig';  % Output layer activation function

%setting initial weight and biases
net.IW{1,1} = -0.5 + 1.*rand(size(net.IW{1,1})); % replace with your desired values
net.b{1} = 0.1.*rand(size(net.b{1}));

% % Divide the dataset into training, validation, and testing sets
net.divideFcn = 'divideind';  % You can use other division functions
net.divideParam.trainInd = 1:0.5*entries_breakpoint; %for training 
net.divideParam.valInd = 1:entries_breakpoint;  % for validating
net.divideParam.testInd = entries_breakpoint:length(Data);  %for testing



% Set training parameters
net.trainParam.epochs = 12000;        % Maximum number of training epochs
% net.trainParam.goal = 10^(-99);         % Performance goal
net.trainParam.max_fail = 100000;        % Maximum number of validation failures
net.trainParam.min_grad = 10^(-9999);     % Minimum gradient for convergence
% net.trainParam.mu_max = 10^(99);
% net.trainParam.mu = 0.9;           % Initial mu (if using 'trainlm' or traingdx)
% net.trainParam.mu_inc = 1;
% net.trainParam.lr = a;           % Learning rate (if using 'trainlm' or traingdx)
% net.trainParam.show = 10;           % Epochs between displays
 net.trainParam.showCommandLine = false;   % Display training progress in command line
% net.trainParam.showWindow = true; % Display training progress in a separate window



% Prepare input time series
inputTimeSeries = tonndata(Data_inputs, false, false);
targetTimeSeries = tonndata(Data_output, false, false);

% Prepare input and target time series with delays
[Data_inputs, inputStates, layerStates, Data_output] = preparets(net, inputTimeSeries, {}, targetTimeSeries);


% Train the NARX neural network
[net, tr] = train(net, Data_inputs, Data_output, inputStates, layerStates);

%%%%%%%%% Make predictions on the training set
predTrainData_outputs = net(Data_inputs, inputStates, layerStates);



predictedOutput = cell2mat(predTrainData_outputs);
expectedOutput = cell2mat(Data_output);
error = (-predictedOutput + expectedOutput)./expectedOutput;


% Plot the actual vs. predicted values
figure('Name', 'NARX model and error (Kriging)');
subplot(2,1,1)
plot(cell2mat(predTrainData_outputs(entries_breakpoint-2:end)'), Color='[0.15,0.81,0.20]', Marker = 'o', MarkerSize=8, LineWidth = lw, LineStyle='none');
hold on;
plot(cell2mat(Data_output(entries_breakpoint-2:end)'), 'k:',LineWidth=lw);
legend('Predicted', 'Expected', 'Position',[0.712738093262627 0.636196107006903 0.17392857340404 0.0754761920202346]);
xlabel('Data Point');
ylabel('Mean Residence (sec)');
title('Actual vs. Predicted Mean Residence Time (Kriging)');
fontname('Times New Roman');

subplot(2,1,2)
    plot(error(entries_breakpoint-2:end)', 'k-')
    ylim(subplot(2,1,2),[-1 0.5])
ylabel("Relative Error")
fontname('Times New Roman');

%%%%
%%%%
% Subplots of MRT against individual variables
figure
subplot(4,1,1)
plot(cell2mat(Data_inputs(entries_breakpoint-2:end)'), cell2mat(predTrainData_outputs(entries_breakpoint-2:end)'), 'm-')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All possible adjustable training parameters:%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 'trainFcn': Specifies the training function.
% Typical values: 'trainlm' (Levenberg-Marquardt), 'trainbfg' (BFGS
% quasi-Newton), 'trainrp' (Rprop), etc. 
% 
% 'trainParam.epochs': Maximum
% number of training epochs.
% % Typical values: 100, 200, 500, etc. 
% 
% 'trainParam.goal': Performance goal.
% Typical values: 1e-6, 1e-5, etc. 
% 
% 'trainParam.max_fail': Maximum number of validation failures. 
% Typical values: 6, 10, etc. 
% 
% 'trainParam.min_grad': Minimum gradient for
% convergence. 
% Typical values: 1e-6, 1e-5, etc. 
% 
% 'trainParam.lr': Learning rate for the
% Levenberg-Marquardt algorithm. 
% Typical values: 0.01, 0.001, etc. (if using 'trainlm') 
% 
% 'trainParam.mu':
% Initial mu (Levenberg-Marquardt parameter). 
% Typical values: 0.01, 0.001, etc. 
% 
% (if using 'trainlm') 'trainParam.show':
% Epochs between displays. 
% Typical values: 10, 25, etc. 
% 
% 'trainParam.showCommandLine': Display
% training progress in command line. 
% Typical values: true or false. 
% 
% 'trainParam.showWindow': Display training
% progress in a separate window.
% Typical values: true or false.
%%

figure;
subplot(3,1,1)
plot(Data(2:end,1), predictedOutput(1,:)', 'b*-');
xlabel('Input 1');
ylabel('Mean Residence Time (sec)');
title('MRT vs. Input 1');
subplot(3,1,2)
plot(Data(2:end,2), predictedOutput(1,:)', 'b*-');
xlabel('Input 2');
ylabel('Mean Residence Time (sec)');
title('MRT vs. Input 2');
subplot(3,1,3)
plot(Data(2:end,3), predictedOutput(1,:)', 'b*-');
xlabel('Input 3');
ylabel('Mean Residence Time (sec)');
title('MRT vs. Input 3');
grid on;


% Possible transfer functions:
% 
% compet - Competitive transfer function.
% elliotsig - Elliot sigmoid transfer function.
% hardlim - Positive hard limit transfer function.
% hardlims - Symmetric hard limit transfer function.
% logsig - Logarithmic sigmoid transfer function.
% netinv - Inverse transfer function.
% poslin - Positive linear transfer function1.
% purelin - Linear transfer function1.
% radbas - Radial basis transfer function1.
% radbasn - Radial basis normalized transfer function1.
% satlin - Positive saturating linear transfer function1.
% satlins - Symmetric saturating linear transfer function1.
% softmax - Soft max transfer function1.
% tansig - Symmetric sigmoid transfer function1.
% tribas - Triangular basis transfer function1.


NARX = cell2mat(predTrainData_outputs((entries_breakpoint-2):end))';
errNARX = error(entries_breakpoint-2:end)';

% mdl = fitlm((Data((entries_breakpoint:end),[1,2,3])),cell2mat(predTrainData_outputs((entries_breakpoint-2:end)))');
% disp(mdl)

PredOutput_test = cell2mat(predTrainData_outputs((entries_breakpoint-2:end)));
expectedOutput_test = expectedOutput((entries_breakpoint-2:end));

mdl2 = fitlm(PredOutput_test,expectedOutput_test);
disp(mdl2)
disp(['Standard deviation or error: ', num2str(std(errNARX))])
disp(['Mean Absolute Error: ', num2str(mean(abs(errNARX)))])

toc
