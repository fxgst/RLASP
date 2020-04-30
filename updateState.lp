% input: current state, move, blocks

% move(c, table, 1).
% block(d). block(a). block(b). block(c). block(e).
% { on(c, b, 0); on(a, e, 0); on(b, d, 0); on(e, table, 0); on(d, a, 0) } = 4.

% Define
location(table).
location(X) :- block(X).
on(X,Y) :- on(X,Y,_).

% introduce new location of X
on(X,Y) :- move(X,Y,T).

% don't allow two different blocks on top of the same block
:- on(X,Y), on(X,Z), Y != Z.

#show on/2.
