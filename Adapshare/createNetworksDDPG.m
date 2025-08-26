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

function [criticNetwork, actorNetwork] = createNetworksDDPG(obsInfo, actInfo)
% Resource Allocation -- Neural Network Setup Script
numObs = prod(obsInfo.Dimension);
numAct = prod(actInfo.Dimension);
%% CRITIC
% Define observation and action paths
obsPath = featureInputLayer(numObs,Name="netOin");
actPath = featureInputLayer(numAct,Name="netAin");

% Define common path: concatenate along first dimension
commonPath = [
    concatenationLayer(1,2,Name="cat")
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(1)
    ];

% Add paths to layerGraph network
criticNetwork = layerGraph(obsPath);
criticNetwork = addLayers(criticNetwork, actPath);
criticNetwork = addLayers(criticNetwork, commonPath);

% Connect paths
criticNetwork = connectLayers(criticNetwork,"netOin","cat/in1");
criticNetwork = connectLayers(criticNetwork,"netAin","cat/in2");

% Convert to dlnetwork object
criticNetwork = dlnetwork(criticNetwork);

% Display the number of weights
summary(criticNetwork)
%% ACTOR
% Create the actor network layers
actorNetwork = [
    featureInputLayer(numObs)
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(numAct) 
    ];

% Convert to dlnetwork object
actorNetwork = dlnetwork(actorNetwork);

% Display the number of weights
summary(actorNetwork)