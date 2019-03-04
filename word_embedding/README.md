

(1) Word2vec starts by hot-encoding each word (n=6, where n is the total number of possible words):

king [1,0,0,0,0,0]
brave [0,1,0,0,0,0]
man [0,0,0,1,0,0]
queen [0,0,0,1,0,0]
beautiful [0,0,0,0,1,0]
woman [0,0,0,0,0,1]


(2) Then, from free texts, extract word (X) and their neighbors (y):
king -> brave.

(3) Feed each (X,y) pair into a 3-layer Neuron Net (NN), with mid-layer 2D (e.g.), softmax, cross-entropy, back prop, as usual. 

(4) This learns a 2D representation of the nD raw vector as follows.
king' [1,1]
brave' [1,2]
man' [1,3]
queen' [5,5]
beautiful' [5,6]
woman' [5,7]

(5) Now think of the product: king * {2D space MATRIX} = king' 
dimension of king: 1-by-n
dimension of the MATRIX: n-by-2
so the MATRIX is a mapping function, its n-th row happens to be the representation of each word (the raw vector is like an Dirac Delta indication function). 




---
Acknowledgement: 
Example from this short video: https://www.youtube.com/watch?v=64qSgA66P-8