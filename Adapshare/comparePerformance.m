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

function comparePerformance(varargin)
% This function takes in string as argument {'DDPGAgent','TD3Agent'}
% and plots the performance comparisons between those respective agents.
%
% Note:
% 1) Make sure that different agents that need to be compared have their
% respective folders with files from different runs.
% 2) Names of the folder need to be passed in as arguments to
% 'comparePerformance' function.
%% Process data for each agent
for i = 1:nargin
    agent(i) = processData(varargin{i});
end

%% Plot mean and standard deviation of average reward values and mean of Episode Q0
figureH = figure('Name','Learning Curve');
axH = axes(figureH);
figureH1 = figure('Name','Episode Q0');
axH1 = axes(figureH1);

for i = 1:length(agent)
    episodeIndex = agent(i).EpisodeIndex;
    meanAverageReward = agent(i).meanAverageReward;
    stdAverageReward = 0.5*agent(i).stdAverageReward;
    meanEpisodeQ0 = agent(i).meanEpisodeQ0;
    stdEpisodeQ0 = 0.5*agent(i).stdEpisodeQ0;
    AvgQ0H = plot(axH1, episodeIndex, meanEpisodeQ0,'DisplayName',varargin{i},'LineWidth',2); % plot mean of Episode Q0
    AvgQ0Color = get(AvgQ0H, 'Color');
    hold(axH1,'on');
    AvgRwdH = plot(axH,episodeIndex, meanAverageReward,'DisplayName',varargin{i},'LineWidth',2); % plot mean of average reward
    AvgRwdColor = get(AvgRwdH, 'Color');
    hold(axH,'on');
    % Arrange data for shading standard deviation
    x = [episodeIndex; flipud(episodeIndex)]; % flipud flips the data from down to up
    y = [meanAverageReward + stdAverageReward; flipud(meanAverageReward - stdAverageReward)];
    fill(axH,x, y ,0.99*AvgRwdColor, 'EdgeAlpha', 0, 'FaceAlpha', 0.4, 'HandleVisibility','off'); % plot standard deviation
    
    y1 = [meanEpisodeQ0 + stdEpisodeQ0; flipud(meanEpisodeQ0 - stdEpisodeQ0)];
    fill(axH1,x, y1 ,0.99*AvgQ0Color, 'EdgeAlpha', 0, 'FaceAlpha', 0.4, 'HandleVisibility','off'); % plot standard deviation
end

title(axH,'Learning curve comparison','FontSize',12)
title(axH1,'Episode Q0 comparison','FontSize',12)

numAxes = [axH,axH1];
for i= 1:length(numAxes)
    grid(numAxes(i),'on');
    xlabel(numAxes(i),'Episode Number')
    ylabel(numAxes(i),'Episode Reward')

    lgd = legend(numAxes(i),'Location','southeast','FontSize',10);
    title(lgd,'AGENTS');
    %legend(numAxes(i),'boxoff');
end
end

function agent = processData(folderName)
% Extract and process data from the saved agent files within specific agent
% folder.
% This function extracts and calculates mean of average reward 
% from all the runs and saves it within agent structure.

% Extract information about all files within the folder 'folderName'
files = dir(folderName);
addpath(folderName); % Add folder to path
files(1:2) = []; % Remove . and .., that automatically gets listed along with filenames

% Consolidate average reward values and Episode Index from different runs
for i = 1:size(files,1)
    S = load(files(i).name);
    if isfield(S,'savedAgentResultStruct')
        res = S.savedAgentResultStruct.TrainingStats;
        info = S.savedAgentResultStruct.Information;
    else
        res = S.savedAgentResult;
        info = res.Information;
    end
    agent.averageReward(:,i) = res.AverageReward;
    agent.EpisodeQ0(:,i) = res.EpisodeQ0;
    agent.EpisodeIndex = res.EpisodeIndex;
    agent.ElapsedTime = info.ElapsedTime;
end

% Calculate mean and std of average reward
agent.meanAverageReward = mean(agent.averageReward,2);
agent.stdAverageReward = std(agent.averageReward,0,2);
agent.meanEpisodeQ0 = mean(agent.EpisodeQ0,2);
agent.stdEpisodeQ0 = std(agent.EpisodeQ0,0,2);
% Remove folder from path
rmpath(folderName);
end