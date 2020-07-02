% enumerate all valid states for the blocks world given some blocks

% input predicates ********************************************************************************
block(a; b; c; d; e).%; f; g; h; i).%; j).
% *************************************************************************************************

% state/1 with state(X) if X is a term representation of a state predicate
state(on(X,Y)) :- on(X,Y).

% the table and every block is a location
location(table).
location(X) :- block(X).

% every block is placed on exactly one location
{ on(X,L) : location(L), L != X  } = 1  :- block(X).

% there cannot be more than one block on any other block
:- block(X), { on(Y,X) : block(Y) } > 1.

% every block has to be supported, directly or indirectly, by the table
supported(X) :- on(X,table).
supported(X) :- on(X,Y), supported(Y).
:- block(X), not supported(X).

#show state/1.
