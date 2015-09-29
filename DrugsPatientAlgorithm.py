'''
    Author: Dominick Moreno
    Directory ID: dmoreno1
    Date: September 22nd, 2014
    Class: ENEE 446
    Assignment: Homework 1, Problem 2
    
    *****************************************************************
    ** NOTE: This script requires Python 2.7 and the NumPy library **
    *****************************************************************
    
    This algorithm attempts to solve the combinatorial problem as discussed
in class. Note the terminology for the patient/drug problem is used instead
of customer/watch. Provided are several hard code matrices that represent
specific drug trials. Output provides a matrix of the same dimensions, where
all nonzero elements in the Pth row indicate the drugs patient P is taking

    It does not attempt to minimize the number of patients. Additionally it
will not make a selection if it will result in an entirely identical set of
combinations.

    In summary this algorithm works by building a ragged array representation
of all of the possible combinations of drugs. In each cell of this array is
a list of all of the rows of the matrix that can potentially make this
combination. Additionally a corresponding structure is built that keeps track
of the number of times each combination has been used

    This algorithm works in 3 steps. Let:
    
        P = number of patients
        D = number of drugs
        M = max number of drugs each patient can take
        
    These steps are:
    
        1. Build the ragged array structure that this algorithm processes - O(P * D^2)
        2. Make initial combination selections. Ie for each patient, choose 2 drugs
        according to specific criteria - O(P^3 * D^2)
        3. Make the remaining selections for each patient - O(M^4 * P * D^2)    
        
    1.    In summary this algorithm works by building a ragged array
    representation of all of the possible combinations of drugs. In each cell
    of this array is a list of all of the rows of the matrix that can
    potentially make this combination. Additionally a corresponding structure
    is built that keeps track of the number of times each combination has been
    used
    
    2.    The combinatorial matrix is iterated over looking for various criteria to
    make its initial selections. In particular making initial combinations out of drugs
    that have not been chosen is prioritized highest (even if they exist in different
    combinations; eg it would try to avoid doing D1D3 if D1D2 were already made). If
    completely impossible to avoid it will allow 1 previous use of a drug, then 2, etc.
    
          Its second priority is to make combinations as unique to a patient as possible.
    That is, all other things equal, if combination D1D2 can be made by P1 and P2, but
    combination D1D3 can only be made by P1 it will choose P1's initial pair to be D1D3.
    
    3.    The remaining selections for each patient are chosen by iterating through the
    combinatorial structure, considering each combination P1 could make, and determining
    if that would result in the maximum number of combinations. What makes this computation
    so heavily dependent on M is that it (1) runs for the remaining number of choices,
    (2) Makes sure the number of choices is maximal, and (3,4) builds all of the potential
    combinations in each iteration to check if that choice is maximal.
    
    There are lots of things that could be done to make this faster and more accurate.
However since there was a very limited amount of time that could be spent on development
and implementation, "good enough" in a novel way was achieved. 
'''

import numpy as np
import itertools
'''
data = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
'''
'''
data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
'''
'''
data = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
'''

data = np.array([[1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
                [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                [1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
                [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1],
                [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
                [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1]])

'''
data = np.array([[1, 1, 1, 1, 1, 0, 0, 0],
                 [0, 0, 0, 1, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1, 1, 0],
                 [1, 1, 0, 0, 0, 1, 1, 1]])
'''
#Get r patients and m drugs, define k max number of drugs to take
numPatients = data.shape[0] #r patients
numDrugs = data.shape[1] #m drugs
maxDrugsPerPatient = 5 #k max drugs

'''
    Initialize the combinatorial and flag matrices
    
    The combinatorial matrix represents every possible drug combination. Note
that there are going to be (m*(m+1))/2 possible combinations. Each cell at
M[i,j] will contain all of the patients the combination DiDj.

    The flag matrix directly corresponds to the combinatorial matrix and
contains the number of times that combination has occured in the construction
of selections of drugs
'''
combMat = []
numOccur = []
for i in range(numDrugs - 1):
    combRow = []
    occurRow = []
    for j in range(numDrugs - 1 - i):
        combRow.append([])
        occurRow.append(0)
        
    combMat.append(combRow)
    numOccur.append(occurRow)
    
#Initialize the matrix that indicates the selections made
selections = np.zeros((numPatients, numDrugs))

'''
    Fill each appropriate cell in combMat with all of the patients who can make
that particular combination
'''
#For each patient (ie row)
for patient in range(numPatients):
    #Make a list of all the nonzero indices (ie all drugs that patient can take)
    nonzeroIndices = np.nonzero(data[patient])[0]
    
    # For each possible combination of drugs
    for combo in (list(d for d in itertools.combinations(nonzeroIndices, 2))):
        #Add this patient to cell in combMat that represents this combination
        combMat[combo[0]][numDrugs - 1 - combo[1]].append(patient)

#For each patient
for patient in range(numPatients):
    matchFoundFlag = False
    
    #For acceptable sums of a column in selections
    for numAcceptableOptions in range(1, numPatients + 1):
        
        #for all acceptable numbers of patients in a cell
        for numAcceptableSelections in (range(0, numPatients)):
            
            #For each col in the drug combinatorial matrix
            for drugA in range(len(combMat)):
                
                #The number of times drugA has already been chosen
                numASelections = len(np.nonzero(selections[:,drugA])[0])
                
                if numASelections > numAcceptableOptions:
                    #This drug has already been used too many times
                    continue

                #For each row (starting at the top) in the drug combinatorial matrix
                for drugB in reversed(range(len(combMat[drugA]))):
                    
                    cell = combMat[drugA][drugB]
                    
                    if patient not in cell:
                        #If this cell doesn't contain this patient, it can't make this combo
                        continue
                    
                    #Number of times drugB has been chosen
                    numBSelections = len(np.nonzero(selections[:,drugB])[0])
                    
                    if numBSelections > numAcceptableOptions:
                        #This drug has already been used too many times
                        continue
                
                    #If an acceptable match is found
                    if len(cell) is numAcceptableOptions:

                        if numOccur[drugA][drugB] is 0:
                            #This combination has not occured yet
                            if max(numASelections, numBSelections) is 0:
                                #A combination has not yet been made that uses A or B
                                matchFoundFlag = True
                                
                            else:
                                if numASelections is 0 and numBSelections <= numAcceptableSelections:
                                    #B has been used but not A
                                    matchFoundFlag = True
                                elif numBSelections is 0 and numASelections <= numAcceptableSelections:
                                    #A has been used but not B
                                    matchFoundFlag = True
                                elif numASelections + numBSelections <= 2*numAcceptableSelections:
                                    #Both have been used but not together
                                    #                OR
                                    #numB/ASelects is over acceptable threshold
                                    matchFoundFlag = True
                        else:
                            #This combination has occurred
                            pass
                        
                        if matchFoundFlag:
                            '''
                            1. Count the number of times this combo has occured (once)
                            2. Indicate drug A for this patient is chosen in selection matrix
                            3. indicate drug B for this patient is chosen in selection matrix
                            '''
                            numOccur[drugA][drugB] = 1
                            selections[patient][drugA] = 1
                            selections[patient][numDrugs - 1 - drugB] = 1
                            
                            break
                        
                    #Selection for this iteration was made, so break to next patient
                    if matchFoundFlag:
                        break
                #Selection for this iteration was made, so break to next patient    
                if matchFoundFlag:
                    break
            #Selection for this iteration was made, so break to next patient    
            if matchFoundFlag:
                break
        
        #Selection for this iteration was made, so break to next patient
        if matchFoundFlag:
            break

#For the remaining choices that need to be made for each patient
for choice in range(3, maxDrugsPerPatient + 1):

    #For each patient
    for patient in range(numPatients):
        matchFound = False
        
        #Get all of the drugs selected for this patient
        drugsSelected = np.nonzero(selections[patient,:])[0]

        '''
        Iterate through combMat looking for the maximum number of new combos
        that can be made
        '''
        for reqNumNewCombos in reversed(range(1, choice)):

            #For each col in the drug combinatorial matrix
            for drugA in range(len(combMat)):

                #For each row (starting at the top) in the drug combinatorial matrix
                for drugB in reversed(range(len(combMat[drugA]))):
                    '''
                    drugB is the index in combMat, not the actual drug
                    drugBSelectionIndx is the actual drug number
                    '''
                    drugBSelectionIndx = numDrugs - 1 - drugB
                    cell = combMat[drugA][drugB]
                    
                    if patient not in cell:
                        #If this cell doesn't contain this patient, it can't make this combo
                        continue
                    
                    if (drugA in drugsSelected) == (drugBSelectionIndx in drugsSelected):
                        '''
                        If neither drugA nor drugB have been selected this is combination
                        is not possible to make with just one more selection. Additionally,
                        if both drugA and drugB have both been selected, there's no reason
                        to make this selection
                        '''
                        continue
                    
                    if drugA in drugsSelected:
                        #Indicate drugB as the drug not already selected
                        newDrug = drugBSelectionIndx
                    else:
                        #Indicate drugA as the drug not already selected
                        newDrug = drugA
                    
                    #Get all of the combos for this patient from those already selected + this one
                    potentialCombos = (list(d for d in itertools.combinations(np.append(drugsSelected, newDrug), 2)))
                    
                    #Count the number of new combinations that would be made if this one were selected
                    comboCount = 0
                    for combo in potentialCombos:
                        
                        '''
                        comboMat is structured so that the lower numbered drug needs to be the first
                        index. This does the swapping if necessary
                        '''
                        if combo[1] > combo[0]:
                            orderedCombo = combo[0], combo[1]
                        else:
                            orderedCombo = combo[1], combo[0]
                        
                        if numOccur[orderedCombo[0]][numDrugs - 1 - orderedCombo[1]] is 0:
                            comboCount += 1
                            
                    #Determine if this potential selection makes enough new combos
                    if comboCount is reqNumNewCombos:
                        #Flag that a selection has been made and make the selection
                        matchFound = True
                        selections[patient][newDrug] = 1
                        
                        #Do swap to index comboMat if necessary
                        for combo in potentialCombos:
                            if combo[1] > combo[0]:
                                orderedCombo = combo[0], combo[1]
                            else:
                                orderedCombo = combo[1], combo[0]
                            
                            #increment the count in all of the appropriate cells
                            numOccur[orderedCombo[0]][numDrugs - 1 - orderedCombo[1]] += 1
                    
                    #Selection for this iteration was made, so break to next patient
                    if matchFound:
                        break
                 
                #Selection for this iteration was made, so break to next patient   
                if matchFound:
                    break
            
            #Selection for this iteration was made, so break to next patient
            if matchFound:
                break

print selections

numCombos = 0
for i in range(len(numOccur)):
    for j in range(len(numOccur[i])):
        if numOccur[i][j] > 0:
            numCombos += 1
                
print "number of distinct combinations: ", numCombos
                    
    
    
    
    
    
    
    