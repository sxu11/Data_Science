
Confounding DAG (directed acyclic graphs), very important:
(1) A -> G -> B
A and B are marginally associated
However, conditioned on G, A and B not associated. 

(2) A <- C <- D <- E -> G -> B 
Same as (fork) E -> D -> C -> A
                 -> G -> B
A and B are associated on the fork                
Condition on E, the path from A to B is blocked.             

(3) A -> G <- B
Information from A and B collide at G. A and B are not associated via this path.
However, conditioned on G, A and B are associated.

A path is d-separated by a set of node C if:
In (1), the middle part is in C
OR
In (2), the middle part is in C
OR
In (3), the middle part is not in C, nor any descent of it. 
Notation: A and B are d-separated by C, A \vertical B | C

Confounder X:
X -> A -> Y
X -> Y 

Frontdoor path: From A to Y that begins out of A 
Causal, do not worry about

Backdoor path: From A to Y through arrows going into A
A <- X -> Y
Must identify a set of variables that block all backdoor paths from treatment to outcome
