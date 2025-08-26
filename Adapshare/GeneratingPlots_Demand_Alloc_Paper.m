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

% Objective: code to generate plots in the paper "AdapShare: An RL-Based Dynamic Spectrum Sharing Solution for O-RAN"
clc;
clear all;
close all;

maxSteps = 100;
gamma = 0:0.01:1;
id = [1,51,101];
% load results
load("td3_comparison.mat");
load("ddpg_comparison.mat");

for i=1:numel(id)
    % plot corresponding to demand and allocation corresponding to N_R = 20
    figure;
    set(gca,'FontSize',40,'FontWeight','bold');
    plot(1:maxSteps,learn_td(1).lte_demand(id(i),1:maxSteps),'-*','LineWidth',2,'MarkerSize',10);
    hold on;
    plot(1:maxSteps,learn_td(1).nr_demand(id(i),1:maxSteps),'-^','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,learn_td(1).lte_alloc(id(i),1:maxSteps),'-+','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,learn_td(1).nr_alloc(id(i),1:maxSteps),'-d','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,learn_ddpg(1).lte_alloc(id(i),1:maxSteps),'-v','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,learn_ddpg(1).nr_alloc(id(i),1:maxSteps),'-x','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,optimal(1).lte_alloc_peak(id(i),1:maxSteps),'-s','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,optimal(1).nr_alloc_peak(id(i),1:maxSteps),'-o','LineWidth',2,'MarkerSize',10);
    hold off;
    grid on
    xlabel('Time Steps','Fontweight','bold','Fontsize',40,'Interpreter','latex');
    ylabel('Allocation/Demand','Fontweight','bold','Fontsize',40,'Interpreter','latex');
    if i==1
        fig_name = sprintf('Nr20_gamma0_alloc.pdf');
    elseif i==2
    
        fig_name = sprintf('Nr20_gamma05_alloc.pdf');
    else
        fig_name = sprintf('Nr20_gamma1_alloc.pdf');
    end
    ylim([0 40]);
    set(gca, 'TickLabelInterpreter', 'latex','FontSize',40);
    set(gcf,'Color','w');
    set(gcf,'Position',get(0,'ScreenSize'));
    legendflex(gca,{'$D_A$','$D_B$','$N^*_A$ (TD3)','$N^*_B$ (TD3)',...
        '$N^*_A$ (DDPG)','$N^*_B$ (DDPG)','$N^*_A$ (Baseline)','$N^*_B$ (Baseline)'},...
        'ncol', 4,'nrow',2,'fontsize',32,'fontweight','bold','Interpreter', 'latex', 'anchor', [2 2], 'buffer', [0 -10]);    
    addpath 'Export_fig'
    cd 'Figures'
    export_fig(fig_name);
    cd '..'
    close;
    
    % plot corresponding to demand and allocation corresponding to N_R = 60
    figure;
    set(gca,'FontSize',40,'FontWeight','bold');
    plot(1:maxSteps,learn_td(2).lte_demand(id(i),1:maxSteps),'-*','LineWidth',2,'MarkerSize',10);
    hold on;
    plot(1:maxSteps,learn_td(2).nr_demand(id(i),1:maxSteps),'-^','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,learn_td(2).lte_alloc(id(i),1:maxSteps),'-+','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,learn_td(2).nr_alloc(id(i),1:maxSteps),'-d','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,learn_ddpg(2).lte_alloc(id(i),1:maxSteps),'-v','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,learn_ddpg(2).nr_alloc(id(i),1:maxSteps),'-x','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,optimal(2).lte_alloc_peak(id(i),1:maxSteps),'-v','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,optimal(2).nr_alloc_peak(id(i),1:maxSteps),'-x','LineWidth',2,'MarkerSize',10);
    hold off;
    grid on
    xlabel('Time Steps','Fontweight','bold','Fontsize',40,'Interpreter','latex');
    ylabel('Allocation/Demand','Fontweight','bold','Fontsize',40,'Interpreter','latex');
    if i==1
        fig_name = sprintf('Nr60_gamma0_alloc.pdf');
    elseif i==2
        fig_name = sprintf('Nr60_gamma05_alloc.pdf');
    else
        fig_name = sprintf('Nr60_gamma1_alloc.pdf');
    end
    ylim([0 40]);
    set(gca, 'TickLabelInterpreter', 'latex','FontSize',40);
    set(gcf,'Color','w');
    set(gcf,'Position',get(0,'ScreenSize'));
   legendflex(gca,{'$D_A$','$D_B$','$N^*_A$ (TD3)','$N^*_B$ (TD3)',...
        '$N^*_A$ (DDPG)','$N^*_B$ (DDPG)','$N^*_A$ (Baseline)','$N^*_B$ (Baseline)'},...
        'ncol', 4,'nrow',2,'fontsize',32,'fontweight','bold','Interpreter', 'latex', 'anchor', [2 2], 'buffer', [0 -10]);    
    addpath 'Export_fig'
    cd 'Figures'
    export_fig(fig_name);
    cd '..'
    close;

    % plot corresponding to demand and allocation corresponding to N_R = 100
    figure;
    set(gca,'FontSize',40,'FontWeight','bold');
    plot(1:maxSteps,learn_td(3).lte_demand(id(i),1:maxSteps),'-*','LineWidth',2,'MarkerSize',10);
    hold on;
    plot(1:maxSteps,learn_td(3).nr_demand(id(i),1:maxSteps),'-^','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,learn_td(3).lte_alloc(id(i),1:maxSteps),'-+','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,learn_td(3).nr_alloc(id(i),1:maxSteps),'-d','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,learn_ddpg(3).lte_alloc(id(i),1:maxSteps),'-v','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,learn_ddpg(3).nr_alloc(id(i),1:maxSteps),'-x','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,optimal(3).lte_alloc_peak(id(i),1:maxSteps),'-v','LineWidth',2,'MarkerSize',10);
    plot(1:maxSteps,optimal(3).nr_alloc_peak(id(i),1:maxSteps),'-x','LineWidth',2,'MarkerSize',10);
    hold off;
    grid on
    xlabel('Time Steps','Fontweight','bold','Fontsize',40,'Interpreter','latex');
    ylabel('Allocation/Demand','Fontweight','bold','Fontsize',40,'Interpreter','latex');
    if i==1
        fig_name = sprintf('Nr100_gamma0_alloc.pdf');
    elseif i==2
        fig_name = sprintf('Nr100_gamma05_alloc.pdf');
    else
        fig_name = sprintf('Nr100_gamma1_alloc.pdf');
    end
    ylim([0 40]);
    set(gca, 'TickLabelInterpreter', 'latex','FontSize',40);
    set(gcf,'Color','w');
    set(gcf,'Position',get(0,'ScreenSize'));
   legendflex(gca,{'$D_A$','$D_B$','$N^*_A$ (TD3)','$N^*_B$ (TD3)',...
        '$N^*_A$ (DDPG)','$N^*_B$ (DDPG)','$N^*_A$ (Baseline)','$N^*_B$ (Baseline)'},...
        'ncol', 4,'nrow',2,'fontsize',32,'fontweight','bold','Interpreter', 'latex', 'anchor', [2 2], 'buffer', [0 -10]);    
    addpath 'Export_fig'
    cd 'Figures'
    export_fig(fig_name);
    cd '..'
    close;
end

% plot corresponding fairness for N_r = 100
figure;
set(gca,'FontSize',40,'FontWeight','bold');
plot(gamma,mean(learn_td(3).fairness,2),'-d','LineWidth',2,'MarkerSize',10);
hold on;
plot(gamma,mean(learn_ddpg(3).fairness,2),'-v','LineWidth',2,'MarkerSize',10);
plot(gamma,mean(optimal(3).fairness_peak,2),'-^','LineWidth',2,'MarkerSize',10);
hold off;
grid on
xlabel('Weighting Factor ($\zeta$)','Fontweight','bold','Fontsize',40,'Interpreter','latex');
ylabel('Fairness Index','Fontweight','bold','Fontsize',40,'Interpreter','latex');
fig_name = sprintf('Fairness_Nr_100.pdf');
ylim([0 1.5]);
set(gca, 'TickLabelInterpreter', 'latex','FontSize',40);
set(gcf,'Color','w');
set(gcf,'Position',get(0,'ScreenSize'));
legendflex(gca,{'$F_{TD3}$','$F_{DDPG}$', '$F_{base}$'},...
    'ncol', 3,'nrow',1,'fontsize',32,'fontweight','bold','Interpreter', 'latex','anchor', [2 2], 'buffer', [0 -10]);
addpath 'Export_fig'
cd 'Figures'
export_fig(fig_name);
cd '..'
close;

% plot corresponding surplus/deficit N_r = 100
figure;
set(gca,'FontSize',32,'FontWeight','bold');
plot(gamma,learn_td(3).sur_def_lte','-*','LineWidth',2,'MarkerSize',10);
hold on;
plot(gamma,learn_td(3).sur_def_nr','-+','LineWidth',2,'MarkerSize',10);
plot(gamma,learn_ddpg(3).sur_def_lte','-v','LineWidth',2,'MarkerSize',10);
plot(gamma,learn_ddpg(3).sur_def_nr','-^','LineWidth',2,'MarkerSize',10);
plot(gamma,optimal(3).sur_def_lte_peak','-s','LineWidth',2,'MarkerSize',10);
plot(gamma,optimal(3).sur_def_nr_peak','-d','LineWidth',2,'MarkerSize',10);
hold off;
grid on
xlabel('Weighting Factor ($\zeta$)','Fontweight','bold','Fontsize',40,'Interpreter','latex');
ylabel('Surplus/Deficit','Fontweight','bold','Fontsize',40,'Interpreter','latex');
fig_name = sprintf('Sur_Def_Nr_100.pdf');
ylim([-1 0.75])
set(gca, 'TickLabelInterpreter', 'latex','FontSize',40);
set(gcf,'Color','w');
set(gcf,'Position',get(0,'ScreenSize'));
legendflex(gca,{'$S_A$ (TD3)','$S_B$ (TD3)', '$S_A$ (DDPG)', '$S_B$ (DDPG)',...
    '$S_A$ (Baseline)','$S_B$ (Baseline)'},...
    'ncol', 3,'nrow',2,'fontsize',32,'fontweight','bold','Interpreter', 'latex', 'anchor', [2 2], 'buffer', [0 -10]);    
addpath 'Export_fig'
cd 'Figures'
export_fig(fig_name);
cd '..'
close;

% plot corresponding fairness for N_r = 20
figure;
set(gca,'FontSize',40,'FontWeight','bold');
plot(gamma,mean(learn_td(1).fairness,2),'-d','LineWidth',2,'MarkerSize',10);
hold on;
plot(gamma,mean(learn_ddpg(1).fairness,2),'-v','LineWidth',2,'MarkerSize',10);
plot(gamma,mean(optimal(1).fairness_peak,2),'-^','LineWidth',2,'MarkerSize',10);
hold off;
grid on
xlabel('Weighting Factor ($\zeta$)','Fontweight','bold','Fontsize',40,'Interpreter','latex');
ylabel('Fairness Index','Fontweight','bold','Fontsize',40,'Interpreter','latex');
fig_name = sprintf('Fairness_Nr_20.pdf');
ylim([0 1.25]);
set(gca, 'TickLabelInterpreter', 'latex','FontSize',40);
set(gcf,'Color','w');
set(gcf,'Position',get(0,'ScreenSize'));
legendflex(gca,{'$F_{TD3}$','$F_{DDPG}$', '$F_{base}$'},...
    'ncol', 3,'nrow',1,'fontsize',32,'fontweight','bold','Interpreter', 'latex','anchor', [2 2], 'buffer', [0 -10]);
addpath 'Export_fig'
cd 'Figures'
export_fig(fig_name);
cd '..'
close;

% plot corresponding surplus/deficit N_r = 20
figure;
set(gca,'FontSize',32,'FontWeight','bold');
plot(gamma,learn_td(1).sur_def_lte','-*','LineWidth',2,'MarkerSize',10);
hold on;
plot(gamma,learn_td(1).sur_def_nr','-+','LineWidth',2,'MarkerSize',10);
plot(gamma,learn_ddpg(1).sur_def_lte','-v','LineWidth',2,'MarkerSize',10);
plot(gamma,learn_ddpg(1).sur_def_nr','-^','LineWidth',2,'MarkerSize',10);
plot(gamma,optimal(1).sur_def_lte_peak','-s','LineWidth',2,'MarkerSize',10);
plot(gamma,optimal(1).sur_def_nr_peak','-d','LineWidth',2,'MarkerSize',10);
hold off;
grid on
xlabel('Weighting Factor ($\zeta$)','Fontweight','bold','Fontsize',40,'Interpreter','latex');
ylabel('Surplus/Deficit','Fontweight','bold','Fontsize',40,'Interpreter','latex');
fig_name = sprintf('Sur_Def_Nr_20.pdf');
ylim([-1 0.75])
set(gca, 'TickLabelInterpreter', 'latex','FontSize',40);
set(gcf,'Color','w');
set(gcf,'Position',get(0,'ScreenSize'));
legendflex(gca,{'$S_A$ (TD3)','$S_B$ (TD3)', '$S_A$ (DDPG)', '$S_B$ (DDPG)',...
    '$S_A$ (Baseline)','$S_B$ (Baseline)'},...
    'ncol', 3,'nrow',2,'fontsize',32,'fontweight','bold','Interpreter', 'latex', 'anchor', [2 2], 'buffer', [0 -10]);    
addpath 'Export_fig'
cd 'Figures'
export_fig(fig_name);
cd '..'
close;

% plot corresponding fairness for N_r = 60
figure;
set(gca,'FontSize',40,'FontWeight','bold');
plot(gamma,mean(learn_td(2).fairness,2),'-d','LineWidth',2,'MarkerSize',10);
hold on;
plot(gamma,mean(learn_ddpg(2).fairness,2),'-v','LineWidth',2,'MarkerSize',10);
plot(gamma,mean(optimal(2).fairness_peak,2),'-^','LineWidth',2,'MarkerSize',10);
hold off;
grid on
xlabel('Weighting Factor ($\zeta$)','Fontweight','bold','Fontsize',40,'Interpreter','latex');
ylabel('Fairness Index','Fontweight','bold','Fontsize',40,'Interpreter','latex');
fig_name = sprintf('Fairness_Nr_60.pdf');
ylim([0 1.25])
set(gca, 'TickLabelInterpreter', 'latex','FontSize',40);
set(gcf,'Color','w');
set(gcf,'Position',get(0,'ScreenSize'));
legendflex(gca,{'$F_{TD3}$','$F_{DDPG}$', '$F_{base}$'},...
    'ncol', 3,'nrow',1,'fontsize',32,'fontweight','bold','Interpreter', 'latex','anchor', [2 2], 'buffer', [0 -10]);
addpath 'Export_fig'
cd 'Figures'
export_fig(fig_name);
cd '..'
close;

% plot corresponding surplus/deficit N_r = 60
figure;
set(gca,'FontSize',32,'FontWeight','bold');
plot(gamma,learn_td(2).sur_def_lte','-*','LineWidth',2,'MarkerSize',10);
hold on;
plot(gamma,learn_td(2).sur_def_nr','-+','LineWidth',2,'MarkerSize',10);
plot(gamma,learn_ddpg(2).sur_def_lte','-v','LineWidth',2,'MarkerSize',10);
plot(gamma,learn_ddpg(2).sur_def_nr','-^','LineWidth',2,'MarkerSize',10);
plot(gamma,optimal(2).sur_def_lte_peak','-s','LineWidth',2,'MarkerSize',10);
plot(gamma,optimal(2).sur_def_nr_peak','-d','LineWidth',2,'MarkerSize',10);
hold off;
grid on
xlabel('Weighting Factor ($\zeta$)','Fontweight','bold','Fontsize',40,'Interpreter','latex');
ylabel('Surplus/Deficit','Fontweight','bold','Fontsize',40,'Interpreter','latex');
fig_name = sprintf('Sur_Def_Nr_60.pdf');
ylim([-1 0.75])
set(gca, 'TickLabelInterpreter', 'latex','FontSize',40);
set(gcf,'Color','w');
set(gcf,'Position',get(0,'ScreenSize'));
legendflex(gca,{'$S_A$ (TD3)','$S_B$ (TD3)', '$S_A$ (DDPG)', '$S_B$ (DDPG)',...
    '$S_A$ (Baseline)','$S_B$ (Baseline)'},...
    'ncol', 3,'nrow',2,'fontsize',32,'fontweight','bold','Interpreter', 'latex', 'anchor', [2 2], 'buffer', [0 -10]);    
addpath 'Export_fig'
cd 'Figures'
export_fig(fig_name);
cd '..'
close;