# Drug-Patient Algorithm Implementation

In the University of Maryland, College Park Electrical Engineering department's ENEE 446: Digital Computer Design, the first (and only programming) assignment was given to us as follows:

A pharmaceutical company wants to market some "m" new drugs but before then, it would like to test them to see if any two of these drugs taken together would cause a problem. To try out as many pairs as possible out of the "m(m-1)/2" choices of pairs, it sets up a beta test on "r" patients, each with her/his own set of drugs that can be administered on her/him.

Let the cardinalities of these sets be denoted by "a(i)", for "i = 1, 2, ..., r", and suppose that the reactions of the "r" patients to the "m" drugs are specified by an "r" by "m" binary matrix, where a 1 in the "i"th row and "j"th column indicates a reaction and a 0 in the same position indicates no reaction to drug "d(j)" by patient "p(i)".

Further suppose that each patient can be administered only "k" drugs in each beta test.

The pharmaceutical company would like to find an optimal assignment of the "m" drugs to the "r" patients so as to maximize the number of pairs of drugs whose pairwise potency can be tested by the "r" patients.

Simply bruteforcing this approach is impossible, you can't just iterate through every possible combination. For the following values:

  m = 20
  k = 5
  a(i) = 10
  i = 1, 2, ..., 20
  r = 20
  
There are 2.52^(20) * 10^(4) possible combinations. If we were able to check 1 calculation per nanosecond, it would take approximately 1.47*10^34 years to check all possible combinations.

Our class assignment was to devise an algorithm that could devise such a patient-drug assignment, given a patient-drug-availability matrix, in a "reasonable" amount of time. The instructor encouraged us to implement a greedy method algorithm, but I decided to take an altogether different approach. You can see my algorithm's description in the beginning comment section, or the accompanying algorithm description document. While effective, for the matrix given to us this algorithm is not entirely complete, it appears to miss 2 potential drug pairs (out of approximately 115), but runs extremely quickly, especially compared to my classmate's greedy algorithms.
  
