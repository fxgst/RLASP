% input: const t, state, goal, blocks


%block(d). block(a). block(b). block(c). block(e).
%on(c, b, 0). on(a, e, 0). on(b, d, 0). on(e, table, 0). on(d, a, 0). 
%subgoal(e, d). subgoal(d, c). subgoal(b, a). subgoal(c, b). subgoal(a, table).
%#const t = 0.


% plan t steps ahead
time(0..t).

% the table and every block is a location
location(table).
location(X) :- block(X).

% has the goal state been reached?
goal(T) :- time(T), { not on(X,Y,T) : subgoal(X,Y) } = 0.    
goal    :- goal(T). 

% a block is occupied if there is something on top of it, otherwise it is free
% the table is always free
occupied(X,T) :- block(X), on(_,X,T).
free(X,T)     :- location(X), time(T), not occupied(X,T).

% at each time point, make one move unless the goal has been reached
{ move(X,Y,T) : block(X), location(Y), free(X,T), free(Y,T), not on(X,Y,T), X != Y } = 1 :- time(T), not goal(T).

% by default, blocks don't change location (frame axiom)
on(X,Y,T+1) :- on(X,Y,T), not -on(X,Y,T+1), time(T).

% effects of action move
-on(X,Y,T+1) :- move(X,_,T), on(X,Y,T), time(T).
 on(X,Y,T+1) :- move(X,Y,T), time(T).

% TODO: 
% either reward based planning: find a plan that maximizes a reward function
% or let agent decide on next steps
reward(1000,T) :- goal(T), not goal(T-1). % big reward for reaching the goal the first time
reward(-1,T)  :- move(_,_,T-1).           % small penalty for each move

totalReward(S) :- S = #sum { R,T : reward(R,T)  }.

#show move/3.
#show totalReward/1.
