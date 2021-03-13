%% Initialization
%  Initialize the world, Q-table, and hyperparameters
clear
close all
clc

setupQlearning

% PARAMS:
WORLD = 4;
MAX_EPISODES = 2000;
GAMMA = 0.6;
LEARN_RATE = 0.1;
epsilon = 0.9;

% init world
s = gwinit(WORLD); 

% init Q:
% cols for each state, row for each action
Q = zeros(s.ysize,s.xsize,4);
Q(1,:,2) = -inf;
Q(:,1,4) = -inf;
Q(s.ysize,:,1) = -inf;
Q(:,s.xsize,3) = -inf;

% gwaction directions
%   2
% 4 1 3

%% Training loop
%  Train the agent using the Q-learning algorithm.

% for each episode
for episode = 1:MAX_EPISODES
    
    % Init a start state s
    s = gwinit(WORLD);
    
    while ~s.isterminal
        [a, ~] = chooseaction(Q, s.pos(1), s.pos(2), [1 2 3 4], 0.25*[1 1 1 1], epsilon);
        s_1 = gwaction(a);
        
        if s_1.isvalid
            r = s_1.feedback;
            V = getvalue(Q);
            Q(s.pos(1), s.pos(2), a) = (1-LEARN_RATE)*Q(s.pos(1), s.pos(2), a) +...
                LEARN_RATE*(r + GAMMA*V(s_1.pos(1), s_1.pos(2)));
            s = s_1;
        else
            % set Q for that state&action to -inf for invalid moves
            Q(s.pos(1),s.pos(2),a) = -99;
        end
    end
    
    % less exploration over time
    epsilon = epsilon*0.999;
end
%% visualize Q, P and V
% figure(1)
% imagesc(Q(:,:,1))
% 
% figure(2)
% imagesc(Q(:,:,2))
% 
% figure(3)
% imagesc(Q(:,:,3))
% 
% figure(4)
% imagesc(Q(:,:,4))
% 
figure(5)
V = getvalue(Q);
imagesc(V)

figure(6)
hold on;
P = getpolicy(Q);
gwinit(WORLD)
gwdraw()
gwdrawpolicy(P)
hold off;

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.

P = getpolicy(Q);
s = gwinit(WORLD);
while ~s.isterminal
    a = P(s.pos(1),s.pos(2));
    s = gwaction(a);
    figure(7)
    gwdraw("Policy",P)
    pause(0.01)
end
