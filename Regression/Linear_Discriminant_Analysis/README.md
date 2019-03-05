
PCA: reduces dimensions by focusing on genes with the most variation
However, we are not interested in those, but maximizing the seperability between the two groups so can make the best decisions. 

LDA: Also reduce dimensions by projecting on a new axis. 
Two criteria:
(1) Maximize the distance between means.
(2) Minimize the variation ("scatter", s^2) w/i each category. 
Optimize:
(mu1-mu2)^2/(s1^2+s2^2)

LDA for 3 categories (for 3 genes):
(1) Find the central for all points
(2) Maximize the distance between each mean and the central.
(3) Minimize the variation w/i each category. 
Also, we are choosing 2 new axes. 

Can we create 3 categories (2 axes) for separating 10,000 genes?

![Alt text](images/LDAvsPCA.png?raw=true "Optional Title")

Similarities of LDA and PCA:
PC1 (1st new axis created by PCA): accounts for the most variation in the data X
LD1 (1st new axis created by LDA): accounts for the most variation between the categories y

---
Reference: https://www.youtube.com/watch?v=azXCzI57Yfc