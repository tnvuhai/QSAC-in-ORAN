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

% Objective: Code corresponding to TD3 agent.

clc;
clear all;
close all;

mdl = "RL_Resource_Allocation";
open_system(mdl)

%Fix the random generator seed for reproducibility.
rng(0);

% Number of past data points to use in observation
numPastDataPoints = 10;
scaling_factor = 0.01;

% Initialize resource demand for each network (to avoid compilation errors)
LTEDemand = 25;
NRDemand = 25;
LTEInitialDemand = zeros(1,10);
NRInitialDemand = zeros(1,10);
LTEDemandSeries = zeros(1,numPastDataPoints+1);
NRDemandSeries = zeros(1,numPastDataPoints+1);

% Set initial resource allocation for simulation
initAllocation = [25 25]*scaling_factor;

% Observation info
obsInfo = rlNumericSpec([1 (2*numPastDataPoints+2)]);
% Name and description are optional and not used by the software
obsInfo.Name = "observations";
obsInfo.Description = "[D_A, D_B, gamma, N_R]";

% Action info
actInfo = rlNumericSpec([1 2],"LowerLimit",0,"UpperLimit",100*scaling_factor); 
actInfo.Name = "Allocation";

% Create the environment object with Fast Restart enabled.
env = rlSimulinkEnv("RL_Resource_Allocation","RL_Resource_Allocation/controller",obsInfo,actInfo,...
    UseFastRestart="on");

% Set a reset function
env.ResetFcn = @(in)networkResetFcn(in,numPastDataPoints);

% Select and Create Agent for Training
AgentSelection = "TD3";
switch AgentSelection
    case "DDPG"
        agent = createDDPGAgent(obsInfo,actInfo);
    case "TD3"
        agent = createTD3Agent(obsInfo,actInfo);
    otherwise
        disp("Assign AgentSelection to DDPG or TD3")
end
Ts = 1;
agent.SampleTime = Ts;

%Specify Training Options and Train Agent
maxEpisodes = 20000;
maxSteps = 1;
evaluator = rlEvaluator(EvaluationFrequency=50,NumEpisodes=1,UseExplorationPolicy=false);
opts = rlTrainingOptions(...
    MaxEpisodes=maxEpisodes,...
    MaxStepsPerEpisode=maxSteps,...
    ScoreAveragingWindowLength=50,...
    Verbose=false,...
    Plots="training-progress",...
    StopTrainingCriteria="EpisodeCount",...
    StopTrainingValue=maxEpisodes,...
    SaveAgentCriteria="EpisodeCount",...
    SaveAgentValue=maxEpisodes,...
    SaveAgentDirectory = pwd + "\" + sprintf("%sAgent",AgentSelection));

% Train the agent using the train function.
doTraining = false;
if doTraining    
    % Train the agent.
    info = train(agent,env,opts);
    % Disable doTraining flag after training is finished
    doTraining = false;
else
    % Load a pretrained agent for the selected agent type.
    if strcmp(AgentSelection,"DDPG")
       load("DDPGAgent/Agent20000_run1.mat","saved_agent")
       agent = saved_agent;
    else strcmp(AgentSelection,"TD3")
       load("TD3Agent/Agent20000_run1.mat","saved_agent")
       agent = saved_agent;
    end
end

% Simulate Trained Agents
numsim = 1;
maxSteps = 500;

% Set gamma and N_R for simulation
gamma = 0:0.01:1;
N_R = [20,60,100];
learn_td = struct();

for n = 1:numel(N_R)
    % initialize values for RL based allocation
    lte_alloc = zeros(numel(gamma),maxSteps);
    nr_alloc = zeros(numel(gamma),maxSteps);
    lte_demand = zeros(numel(gamma),maxSteps);
    nr_demand = zeros(numel(gamma),maxSteps);
    sur_def_lte = zeros(numel(gamma),1);
    sur_def_nr = zeros(numel(gamma),1);
    fairness = zeros(numel(gamma),maxSteps);
    for i = 1:numel(gamma)
        % Select which data file to use for simulation
        LTEFile = "Data/LTE_Demand_0.xlsx";
        NRFile = "Data/NR_Demand_0.xlsx";
        set_param("RL_Resource_Allocation/LTE Data/Simulation Data/Timeseries Data",...
            "FileName",LTEFile);
        set_param("RL_Resource_Allocation/NR Data/Simulation Data/Timeseries Data",...
            "FileName",NRFile);
        
        % Find initial demand values for each network (to "warm start" scheduler)
        LTEDataTimeseries = readtable(LTEFile);
        NRDataTimeseries = readtable(NRFile);
        
        LTEInitialDemand = LTEDataTimeseries.NRB(10:-1:1);
        NRInitialDemand = NRDataTimeseries.NRB(10:-1:1);
        
        env.UseFastRestart = "off"; 
        env.ResetFcn = @(in)networkResetFcnSim(in,numPastDataPoints,N_R(n),gamma(i));
        
        simout = sim(agent,env,rlSimulationOptions("MaxSteps",maxSteps,"NumSimulations",numsim));
       
        alloc = simout.Action.Allocation.Data;
        for j=1:maxSteps
            lte_alloc(i,j) = alloc(1,1,j)*100;
            nr_alloc(i,j) = alloc(1,2,j)*100;
        end
        lte_demand(i,:) = LTEDataTimeseries.NRB(1:maxSteps);
        nr_demand(i,:) = NRDataTimeseries.NRB(1:maxSteps);
        sur_def_lte(i,1) = mean((lte_alloc(i,:) - lte_demand(i,:))./lte_demand(i,:));
        sur_def_nr(i,1) = mean((nr_alloc(i,:) - nr_demand(i,:))./nr_demand(i,:));
        fairness(i,:) = 0.5*(((lte_alloc(i,:) + nr_alloc(i,:)).^2)./(lte_alloc(i,:).^2 + nr_alloc(i,:).^2));
    end
    learn_td(n).lte_alloc = lte_alloc;
    learn_td(n).nr_alloc = nr_alloc;
    learn_td(n).lte_demand = lte_demand;
    learn_td(n).nr_demand = nr_demand;
    learn_td(n).sur_def_lte = sur_def_lte;
    learn_td(n).sur_def_nr = sur_def_nr;
    learn_td(n).fairness = fairness;
end

% initialize structure to save values for optimal allocation based on
% demand statistics
optimal = struct();

% compute demand statistics
demand_lte = (LTEDataTimeseries.NRB(1:maxSteps))';
demand_nr = (NRDataTimeseries.NRB(1:maxSteps))';
mu_lte =  mean(demand_lte);
mu_nr = mean(demand_nr);
var_lte = var(demand_lte);
var_nr = var(demand_nr);
peak_lte = max(demand_lte);
peak_nr = max(demand_nr);

% initialize values for the optimization problem
% set parameters for MATLAB solver
options = optimset('Algorithm','interior-point','TolX',1e-10,...
         'TolFun',1e-8,'TolCon',1e-10,'MaxFunEval',1e6,'MaxIter',1e6,'Display','off');

for n=1:numel(N_R)
    % initialize values for maximum demand based resource allocation
    lte_alloc_peak = zeros(numel(gamma),maxSteps);
    nr_alloc_peak = zeros(numel(gamma),maxSteps);
    lte_demand_peak = zeros(numel(gamma),maxSteps);
    nr_demand_peak = zeros(numel(gamma),maxSteps);
    sur_def_lte_peak = zeros(numel(gamma),1);
    sur_def_nr_peak = zeros(numel(gamma),1);
    fairness_peak = zeros(numel(gamma),maxSteps);
    
    % initialize values for average demand based resource allocation
    lte_alloc_avg = zeros(numel(gamma),maxSteps);
    nr_alloc_avg = zeros(numel(gamma),maxSteps);
    lte_demand_avg = zeros(numel(gamma),maxSteps);
    nr_demand_avg = zeros(numel(gamma),maxSteps);
    sur_def_lte_avg = zeros(numel(gamma),1);
    sur_def_nr_avg = zeros(numel(gamma),1);
    fairness_avg = zeros(numel(gamma),maxSteps);
    
    % optimal values of the optimization value with linear inequality constraint
    N_R_init = zeros(1,2);
    lb = zeros(1,2);
    ub = [N_R(n) N_R(n)];
    
    % finding the optimal value using fmincon with mean as input 
    for g = 1:numel(gamma)
        [N_avg,fval_avg,exitflag_avg] = fmincon(@(N)optim_allocation_avg(N,...
            gamma(g),mu_lte,mu_nr,var_lte,var_nr),N_R_init,[1 1],N_R(n),[],[],lb,ub,[],options);
        lte_alloc_avg(g,:) = N_avg(1);
        nr_alloc_avg(g,:) = N_avg(2);
        % compute the surplus/deficit seen by LTE and NR at optimal resource allocation
        sur_def_lte_avg(g,1) = mean((lte_alloc_avg(g,:) - demand_lte)./demand_lte);
        sur_def_nr_avg(g,1) = mean((nr_alloc_avg(g,:) - demand_nr)./demand_nr);
        % compute the Jain's fairness index corresponding to the optimal resource allocation
        fairness_avg(g,:) = 0.5*(((lte_alloc_avg(g,:) + nr_alloc_avg(g,:)).^2)./(lte_alloc_avg(g,:).^2 + nr_alloc_avg(g,:).^2));
    end
    for g = 1:numel(gamma)
        [N_peak,fval_peak,exitflag_peak] = fmincon(@(N)optim_allocation_peak(N,gamma(g),ceil(peak_lte),ceil(peak_nr)),N_R_init,[1 1],N_R(n),[],[],lb,ub,[],options);
        lte_alloc_peak(g,:) = N_peak(1);
        nr_alloc_peak(g,:) = N_peak(2);
        % compute the surplus/deficit seen by RAN A and RAN B at optimal resource allocation
        sur_def_lte_peak(g,1) = mean((lte_alloc_peak(g,:) - demand_lte)./demand_lte);
        sur_def_nr_peak(g,1) = mean((nr_alloc_peak(g,:) - demand_nr)./demand_nr);
        % compute the Jain's fairness index corresponding to the optimal resource allocation
        fairness_peak(g,:) = 0.5*(((lte_alloc_peak(g,:) + nr_alloc_peak(g,:)).^2)./(lte_alloc_peak(g,:).^2 + nr_alloc_peak(g,:).^2));
    end
    optimal(n).lte_alloc_avg = lte_alloc_avg;
    optimal(n).lte_alloc_peak = lte_alloc_peak;
    optimal(n).nr_alloc_avg = nr_alloc_avg;
    optimal(n).nr_alloc_peak = nr_alloc_peak;
    optimal(n).sur_def_lte_avg = sur_def_lte_avg;
    optimal(n).sur_def_nr_avg = sur_def_nr_avg;
    optimal(n).sur_def_lte_peak = sur_def_lte_peak;
    optimal(n).sur_def_nr_peak = sur_def_nr_peak;
    optimal(n).fairness_avg = fairness_avg;
    optimal(n).fairness_peak = fairness_peak;
end
save("td3_comparison.mat","learn_td","optimal");