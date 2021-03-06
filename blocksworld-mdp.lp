% MDP specification for the blocks world

% input predicates ********************************************************************************
% current(X) ...	X is a term representation of a state predicate, provided by BlocksWorld.py, optional
% action(X) ...  	X is term representation of an action that has been chosen, provided by BlocksWorld.py, optional
% subgoal(X,Y) ... 	X is on top of Y in goal state
% constant horizon representing the planning horizon

subgoal(a, table). subgoal(b, a). subgoal(c, b). subgoal(d, c).% subgoal(e, d). subgoal(f, e). subgoal(g, f). subgoal(h, g). subgoal(i, h). subgoal(j, i). subgoal(k, j). subgoal(l, k).
% *************************************************************************************************

#defined action/1. % silence warning that action does not appear in rule head

% executable(X) ... X is term representation of an action that is executable in the follow-up state
executable(move(X,Y)) :- executable(X,Y,1), horizon > 0.
executable(move(X,Y)) :- executable(X,Y,0), horizon = 0.

on(X,Y,0) :- current(on(X,Y)).
move(X,Y,0) :- action(move(X,Y)).

% state(X) ... X is a term representation of a state predicate that holds in the follow-up state
state(on(X,Y)) :- on(X,Y,1).

% nextReward(N) ... N is the reward the agents gets after performing a given action
nextReward(N) :- totalReward(N,1). 

% bestAction(X) ... X is a term representation of the next action that yields maximal reward within horizon steps
bestAction(move(X,Y)) :- move(X,Y,0).

% consider horizon planning steps, provided by BlocksWorld.py
time(0..horizon).

goal(T) :- time(T), { not on(X,Y,T) : subgoal(X,Y) } = 0.    
 
% rewards depending on the current state, the chosen action, and the follow-up state
reward(100,T) :- goal(T), not goal(T-1).        % big reward for reaching the goal the first time
reward(-1,T)  :- move(_,_,T-1).                 % small penalty for each move

% Alternatively, consider rewards for reaching subgoals and penalties for loosing them
%reward(100,T)  :- subgoal(X,Y), on(X,Y,T), not on(X,Y,T-1).    % reward for reaching a subgoal the first time
%reward(-100,T) :- subgoal(X,Y), on(X,Y,T-1), not on(X,Y,T).    % penalty for loosing a subgoal
%reward(-1,T)   :- move(_,_,T-1).                               % small penalty for each move

% background knowledge ****************************************************************************

% the table and every block is a location
block(X) :- on(X,_,0).

location(table).
location(X) :- block(X).

occupied(X,T) :- block(X), on(_,X,T).
free(X,T)     :- location(X), time(T), not occupied(X,T).

% at each time point, make one move unless the goal has been reached
executable(X,Y,T) :- block(X), free(X,T), free(Y,T), not on(X,Y,T), X != Y, not goal(T).
{ move(X,Y,T) : executable(X,Y,T) } = 1 :- time(T), T < horizon, not goal(T).

% by default, blocks do not change location (frame axiom)
on(X,Y,T+1) :- on(X,Y,T), not -on(X,Y,T+1), time(T).

% effects of action move
-on(X,Y,T+1) :- move(X,_,T), on(X,Y,T), time(T).
 on(X,Y,T+1) :- move(X,Y,T), time(T).

% maximize the reward function
totalReward(S,T) :- time(T), S = #sum { R : reward(R,T) }.
maxReward(S)     :- S = #sum { R,T : reward(R,T) }.
#maximize { S : maxReward(S) }.

% output signature
#show executable/1.
#show state/1.
#show nextReward/1.
#show bestAction/1.
#show maxReward/1.
