% generate all valid states for the blocks world with n blocks
% blocks can be on the table or stacked on top of each other

% block(X)     ... X is a block
% location(X)  ... X is a location to place a block on
% on(X,Y)      ... block X is placed on location Y
% supported(X) ... block X is directly or indirectly supported by the table    

% the table and every block is a location
location(table).
location(X) :- block(X).

% every block is placed on exactly one location.
{ on(X,L) : location(L), L <> X  } = 1  :- block(X).

% there cannot be more than one block on any other block
:- block(X), { on(Y,X) : block(Y) } > 1.

% every block has to be supported, directly or indirectly, by the table
supported(X) :- on(X,table).
supported(X) :- on(X,Y), supported(Y).
:- block(X), not supported(X).

% hide everything except on/2 in the answer sets
#show on/2.
