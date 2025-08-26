% Disclaimer: NIST-developed software is provided by NIST as a public service. You may use, copy, and distribute copies of the software in any medium, 
% provided that you keep intact this entire notice. You may improve, modify, and create derivative works of the software or any portion of 
% the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed 
% the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards 
% and Technology as the source of the software. 
% 
% NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT, OR ARISING BY 
% OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, 
% AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY 
% DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING 
% BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
% 
% You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated 
% with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, 
% programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a 
% failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection 
% within the United States.

function in = networkResetFcnSim(in,numPastDataPoints,N_R,gamma)

in = setVariable(in,"gamma",gamma);
in = setVariable(in,"N_R",N_R);

a = 0; 
b = N_R/2;
in = setVariable(in,'x0',a+(b-a).*rand(2,numPastDataPoints),'Workspace',"RL_Resource_Allocation");

addpath('Data\')

% Create cell arrays containing data for each network
LTEFiles = cell(1,35);
NRFiles = cell(1,35);
for i = 1:35
    LTEFiles{i} = sprintf('LTE_Demand_%i.xlsx',i-1);
    NRFiles{i} = sprintf('NR_Demand_%i.xlsx',i-1);
end

% Randomly select which CSV pair to pull data from for this episode
idxData = randi([1 numel(LTEFiles)], [1 1]);

LTEData = LTEFiles{idxData}; % LTEData will be a string containing the file name
NRData = NRFiles{idxData};    % NRData will be a string containing the file name

LTEDataTimeseries = readtable(LTEData);
NRDataTimeseries = readtable(NRData);

LTEInitialDemand = LTEDataTimeseries.NRB(10:-1:1);
NRInitialDemand = NRDataTimeseries.NRB(10:-1:1);

in = setVariable(in,"LTEDemandSeries",LTEInitialDemand);
in = setVariable(in,"NRDemandSeries",NRInitialDemand);

dataIndex = randi([(numPastDataPoints+1) height(LTEDataTimeseries)],[1 1]);

LTEDemandSeries = LTEDataTimeseries.NRB(dataIndex:-1:(dataIndex-numPastDataPoints));
NRDemandSeries = NRDataTimeseries.NRB(dataIndex:-1:(dataIndex-numPastDataPoints));

in = setVariable(in,"LTEDemandSeries",LTEDemandSeries);
in = setVariable(in,"NRDemandSeries",NRDemandSeries);