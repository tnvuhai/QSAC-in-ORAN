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

function agent = createDDPGAgent(obsInfo, actInfo)
% Resource Allocation -- DDPG Agent Setup Script
%% Create the actor and critic networks using the createNetworks helper function
[criticNetwork,actorNetwork] = createNetworksDDPG(obsInfo,actInfo);
%% Specify options for the critic and actor representations using rlOptimizerOptions
criticOptions = rlOptimizerOptions('Optimizer','adam','LearnRate',2e-3,... 
                                        'GradientThreshold',1,'L2RegularizationFactor',5e-4);
actorOptions = rlOptimizerOptions('Optimizer','adam','LearnRate',1e-4,...
                                       'GradientThreshold',1,'L2RegularizationFactor',1e-5);
%% Create critic and actor representations using specified networks and options
critic = rlQValueFunction(criticNetwork,obsInfo,actInfo,...
    ObservationInputNames="netOin",ActionInputNames="netAin");
actor  = rlContinuousDeterministicActor(actorNetwork,obsInfo,actInfo);
%% Specify DDPG agent options
agentOptions = rlDDPGAgentOptions;
agentOptions.SampleTime = 1;
agentOptions.DiscountFactor = 0.99;
agentOptions.MiniBatchSize = 48;
agentOptions.ExperienceBufferLength = 1e6;
agentOptions.TargetSmoothFactor = 1e-4;
agentOptions.NoiseOptions = rl.option.GaussianActionNoise;
agentOptions.NoiseOptions.StandardDeviationMin = 0;
agentOptions.NoiseOptions.StandardDeviation = 0.1;
agentOptions.NoiseOptions.StandardDeviationDecayRate = 1e-5;
agentOptions.ActorOptimizerOptions = actorOptions;
agentOptions.CriticOptimizerOptions = criticOptions;

%% Create agent using specified actor representation, critic representation and agent options
agent = rlDDPGAgent(actor,critic,agentOptions);