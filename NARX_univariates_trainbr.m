%% Mean Residence NARX ANN Model coupled with various Interpolation tools 
clear all
clc 
close all
% Load and preprocess your numerical data
A = zeros(5,1);
B = zeros(5,1);
C = zeros(5,1);
D = zeros(5,1);

tic 

% for i = 1:5 
for L = 10       %For comparing 4 different interpolation ratios
Data = load("TSWG_MeanRes_Dataset.txt"); % L/S	Screw Speed (rpm)	Powder flow rate (g/h)	Mean Residence Time (s)
DataTest2_NonInterpolated = Data(1:end, :);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%interpolation ratio and line width   
ratio = 1/L;
lw = 1.4/(1^1.5);

MLP_or_NARX = 3;        % for the plotting of all variables

% Key ANN Parameters
iterations = 1000; %number of training iterations 
cut = 0.50; % traing:testing data cut off point
h = [5,5] ; % hidden layer size. h = [5,5] produced some good results
a = 0.007; % Learning rate a
trainingAlgorithm = 'trainbr';  % the training algorithm. 'trainbr' is absolutely the best!!!!
                                 % (traingdx does LR automatically)
IntMethod = 'Cub';          %Replace this with whichever function abbreviation (Mak, Lin, Cub, or Spl)
InputDelay = 1:1; 
feebackDelay = 1:2;         % [1:1, 1:2] is optimal Modify the architecture as needed


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% method and colours: 
switch IntMethod
    case 'Mak'
        method = 'makima'
        colour = '[0.82,0.00,1.00]'
        Title = 'Actual vs. Expected Mean Residence Time (Makima)';
    case 'Lin'
        method = 'linear'
        colour = '[0.87,0.88,0.00]'
        Title = 'Actual vs. Expected Mean Residence Time (Linear)';
    case 'Cub'
        method = 'cubic'
        colour = 'b'
        Title = 'Actual vs. Expected Mean Residence Time (Cubic)';
    case 'Spl'
        method = 'spline'
        colour = '[0.95,0.52,0.00]'
        Title = 'Actual vs. Expected Mean Residence Time (Cubic Spline)';
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%% Interpolate Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [1:length(Data)]';

F = griddedInterpolant(x,Data, method);
qx = 1:ratio:length(Data);

Vq = F(qx);
Data_Int_Int = Vq;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% data generation %%%%%%%%%%%%%%%%%%%%%%%%%%%
Data = Data_Int_Int;
% Data = normalize(Data);
[entries,attributes] = size(Data);
entries_breakpoint = round(entries*cut); %this is cutting out % of entries
Data_inputs = Data(:,[1:3]); 
Data_output = Data(:, 4);  
Data_inputs_plotting = Data_inputs;

%to compare known with interpolated:
NonInterpPlot_output = NaN(length(Data),1);
for i = 1:length(DataTest2_NonInterpolated)
    NonInterpPlot_output(L*i-L+1,1) = DataTest2_NonInterpolated(i,4); 
end


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


%% Create a NAR neural network
net = narxnet(InputDelay, feebackDelay, h,'open', trainingAlgorithm);  % [1:1, 1:2] is optimal Modify the architecture as needed
                                               %narxnet(1:numInputDelays, 1:numFeedbackDelays, hiddenlayersize(50), feedbackMode, trainFunction)
                                               %  See also TRAINGDM, TRAINGDA, TRAINGDX, TRAINLM, TRAINRP,
                                               %                     TRAINCGF, TRAINCGB, TRAINBFG, TRAINCGP, TRAINOSS.
            
% Configure the network
net.layers{1}.transferFcn = 'tansig';  % You can choose other activation functions
net.layers{2}.transferFcn = 'tansig';  % Output layer activation function

% setting initial weight and biases
    % net.IW{1,1} = 1+0.001.*rand(size(net.IW{1,1})); % replace with your desired values
    % net.IW{1,2} = 1+0.001.*rand(size(net.IW{1,2})); % replace with your desired values
    % net.b{1} = 1+0.001.*rand(size(net.b{1}));

% % Divide the dataset into training, validation, and testing sets
net.divideFcn = 'divideind';  % divideint will spread test and validation 
                              % throughout. divideind will segregate with 
                              % respect to the time series
net.divideParam.trainInd = 1:0.5*entries_breakpoint; %for training 
% net.divideParam.valInd = 1:entries_breakpoint;
% net.divideParam.testInd =  1:entries_breakpoint;  %for testing
net.divideParam.valInd = 1:entries_breakpoint;  % for validating
net.divideParam.testInd =  entries_breakpoint:length(Data);  %for testing




% Set training parameters
net.trainParam.epochs = iterations;        % Maximum number of training epochs
% net.trainParam.goal = 10^(-99);         % Performance goal
net.trainParam.max_fail = 100000;        % Maximum number of validation failures
net.trainParam.min_grad = 10^(-9999);     % Minimum gradient for convergence
net.trainParam.mu_max = 10^(99);
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

% Plot the training, validation, and testing performance
% figure;
% plotperform(tr);


% Plot the actual vs. predicted values
figure('Name', 'model and error');
subplot(2,1,1)
plot(cell2mat(predTrainData_outputs(entries_breakpoint-2:end)'), Color=colour, Marker='o', MarkerSize=8, LineWidth=lw, LineStyle='none');
hold on;
plot(cell2mat(Data_output(entries_breakpoint-2:end)'), 'k-',LineWidth=1);
% hold on;
% plot(NonInterpPlot_output(entries_breakpoint-2:end), 'r*', Linewidth=1);
legend('Predicted', 'Expected', 'Position',[0.717023807548341 0.697148487959284 0.17392857340404 0.0754761920202346]);
title(Title)
xlabel('Data Point');
ylabel('Mean Residence (sec)');
fontname('Times New Roman');

subplot(2,1,2)
    plot(error(entries_breakpoint-2:end)', 'k-')
    % ylim(subplot(2,1,2),[-1 0.5])
ylabel("Relative Error")
fontname('Times New Roman');



%disp(['Mean Absolute Error: ', num2str(mean(abs(error)))])
%disp(['Standard deviation or error: ', num2str(std(error))])

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % Store error value
% range_error = abs(max(error))+abs(min(error));
% 
% error = error(entries_breakpoint-2:end);
% errorRangeVals(L) = range_error';
% errorVects(:,L) = error';
% 
% fprintf("Error of \a=%.0f", a, range_error')
% fprintf("Error Vector of \a=%.0f", a, error')
% end
% 
% lw=1.5;
% %final plots
% figure('Name', 'Change in error range for different learning rate')
% plot(errorRangeVals)
% xlabel('learning rate value (0.00012 to 120')
% ylabel('Error range')
% xticks("manual")
% 
% figure('Name',"Error of output for various learn rates")
%     plot(errorVects(:,1),'LineWidth',lw); hold on
%         plot(errorVects(:,2),'LineWidth',lw); hold on
%             plot(errorVects(:,3),'LineWidth',lw); hold on
%                 plot(errorVects(:,4),'LineWidth',lw); hold on
%                     plot(errorVects(:,5),'LineWidth',lw); hold on
%                          plot(errorVects(:,6),'LineWidth',lw); hold on
% hold off
% legend("a=0.00081","a=0.0081","a=0.081","a=0.81","a=8.1", "81")
% ylabel("error")
% xlabel("sample")
% 
% LEV = length(errorVects);
% figure('Name',"Error of output for various learn rates (Second Portion)")
%     plot(errorVects(11:22,1),'LineWidth',lw); hold on
%         plot(errorVects(11:22,2),'LineWidth',lw); hold on
%             plot(errorVects(11:22,3),'LineWidth',lw); hold on
%                 plot(errorVects(11:22,4),'LineWidth',lw); hold on
%                     plot(errorVects(11:22,5),'LineWidth',lw); hold on
%                          plot(errorVects(11:22,6),'LineWidth',lw); hold on
% hold off
% legend("a=0.00081","a=0.0081","a=0.081","a=0.81","a=8.1", "81") %0.0007, 0.005, 0.01, 0.05, 0.1, 3
% ylabel("error")
% xlabel("sample")
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
plot(Data_inputs_plotting(MLP_or_NARX:end,1), predictedOutput(1,:)', 'b*-');
xlabel('Input 1');
ylabel('Mean Residence Time (sec)');
title('MRT vs. Input 1');
subplot(3,1,2)
plot(Data_inputs_plotting(MLP_or_NARX:end,2), predictedOutput(1,:)', 'b*-');
xlabel('Input 2');
ylabel('Mean Residence Time (sec)');
title('MRT vs. Input 2');
subplot(3,1,3)
plot(Data_inputs_plotting(MLP_or_NARX:end,3), predictedOutput(1,:)', 'b*-');
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


% PredTestData = cell2mat(predTrainData_outputs((entries_breakpoint-2:end)));
% mdl = fitlm(Data(entries_breakpoint:end,[2,4,5,6]), cell2mat(predTrainData_outputs((entries_breakpoint-2):end))');
% disp(mdl)

% 
% figure
% plot(NARX_4_out); hold on
% plot(NARX_10_out); hold on
% plot(NARX_20_out); hold on
% plot(NARX_40_out); hold on
% plot(NARX_100_out); hold off
% legend('H = 4', 'H = 10', 'H = 20', 'H = 40', 'H = 100')
% ylabel('SOC')
% xlabel('Sample')
% title('Change in LSTM Output for Different Hidden Layer Sizes')


% A(i,1) = mdl2.RMSE;
% B(i,1) = mdl2.Rsquared.Adjusted;
% C(i,1) = vpa(std(errNARX),7);
% D(i,1) = vpa(mean(abs(errNARX)),7);
% 
% 
% 
% 
end 
% end 

toc

%%
%%%%%%%%%%%%%%%%%%%%%% Plotting wrt each variable %%%%%%%%%%%%%%%%%%%%%%%%%
%% Uncomment everything below if you want each variable plotted individually

% %% FIxed L/S and SSpeed
% % Define the ranges for X1 and X2
% data = Data;
% range1 = [0.95, 1.05];  % Example range for X1
% range2 = [95,115]; % Example range for X2
% range3 = [48, 53];
% 
% % Find the rows where X1 is within range1
% rows_X1 = find(data(:,1) >= range1(1) & data(:,1) <= range1(2));
% 
% % Find the rows where X2 is within range2
% rows_X2 = find(data(:,2) >= range2(1) & data(:,2) <= range2(2));
% 
% % Find the rows that satisfy both conditions (X1 in range1 AND X2 in range2)
% rows_combined = intersect(rows_X1, rows_X2);
% 
% % Display the rows
% disp('Rows that meet the criteria:');
% disp(rows_combined);
% 
% % Extract the corresponding data points (optional)
% selected_data = data(rows_combined, :); 
% figure
% plot(selected_data(:,4),'k*-')
% % Title('Change in MRT with Fixed L/S')
% ylabel('Mean Residence Time (sec)', FontSize=12)
% xlabel('Feed Flow Rate', FontSize=12)
% fontname('Times New Roman');
% 
% %% fixed L/S and FFRate
% 
% % Find the rows where X1 is within range1
% rows_X1 = find(data(:,1) >= range1(1) & data(:,1) <= range1(2));
% 
% % Find the rows where X2 is within range2
% rows_X3 = find(data(:,3) >= range3(1) & data(:,3) <= range3(2));
% 
% % Find the rows that satisfy both conditions (X1 in range1 AND X2 in range2)
% rows_combined = intersect(rows_X1, rows_X3);
% 
% % Display the rows
% disp('Rows that meet the criteria:');
% disp(rows_combined);
% 
% % Extract the corresponding data points (optional)
% selected_data = data(rows_combined, :); 
% figure
% plot(selected_data(:,4),'k*-')
% % title('Change in MRT with Fixed L/S ratio and Feed Flowrate')
% ylabel('Mean Residence Time (sec)', FontSize=12)
% xlabel('Screw Speed', FontSize=12)
% fontname('Times New Roman');
% 
% 
% %% fixed SSpeed and FFRate
% 
% % Find the rows where X1 is within range1
% rows_X2 = find(data(:,2) >= range2(1) & data(:,2) <= range2(2));
% 
% % Find the rows where X2 is within range2
% rows_X3 = find(data(:,3) >= range3(1) & data(:,3) <= range3(2));
% 
% % Find the rows that satisfy both conditions (X1 in range1 AND X2 in range2)
% rows_combined = intersect(rows_X2, rows_X3);
% 
% % Display the rows
% disp('Rows that meet the criteria:');
% disp(rows_combined);
% 
% % Extract the corresponding data points (optional)
% selected_data = data(rows_combined, :); 
% figure
% plot(selected_data(:,4),'k*-')
% % title('Change in MRT with Fixed Screw Speed and Feed Flowrate')
% ylabel('Mean Residence Time (sec)', FontSize=12)
% xlabel('L/S ratio', FontSize=12)
% fontname('Times New Roman');
% 
% 
% %% fixed L/S, vary SS*FFR
% 
% % Find the rows where X1 is within range1
% rows_X1 = find(data(:,1) >= range1(1) & data(:,1) <= range1(2));
% 
% % Extract the corresponding data points (optional)
% selected_data = data(rows_X1, :); 
% 
% % Create interaction variable SS*FFR
% SSFFR = selected_data(:, 2).*selected_data(:,3)
% 
% figure
% plot(selected_data(:,4),'k*-')
% title('Change in MRT with Fixed L/S ratio and varying Interaction Variable (SS \cdot FFR)')
% ylabel('Mean Residence Time (sec)', FontSize=12)
% xlabel('SS \cdot FFR', FontSize=12)
% fontname('Times New Roman');

