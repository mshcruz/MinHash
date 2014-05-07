#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include "intMatrix.h"
#include "wordsList.h"
#include "setList.h"


#define NUM_HASH_FUNCTIONS 500

void
InitializeSignatureMatrix(intMatrix **aSignatureMatrix, int numSets, int numHashes)
{
  int i, j;
  intList *maxIntValueForSets = NULL;

  for (i = 0; i < numHashes; i++) {
    for (j = 0; j < numSets; j++) {
      AddIntListItem(&maxIntValueForSets, INT_MAX);
    }
    AddListToMatrix(aSignatureMatrix, maxIntValueForSets);
    ClearIntList(&maxIntValueForSets);
  }
}

wordsList*
BuildWordsList(setList *aSetList)
{
  wordsList *wordsList = NULL;
  setList *tmpSetList = aSetList;  

  while (tmpSetList != NULL) {
    set *tmpSet = tmpSetList -> set;
    while (tmpSet != NULL) {
      AddWord(&wordsList, tmpSet -> elementValue);
      tmpSet = tmpSet -> next;
    }
    tmpSetList = tmpSetList -> next;
  }
  
  return wordsList;
}

intMatrix*
BuildCharacteristicMatrix(setList *aSetList, wordsList *aWordsList)
{
  intMatrix *characteristicMatrix = NULL;
  setList *tmpSetList = NULL;
  set *tmpSet = NULL;
  intList *wordBelongsToSets = NULL;
  wordsList *tmpWordsList = aWordsList;
  
  //For each unique word in the universal set
  while (tmpWordsList != NULL) {
    tmpSetList = aSetList;
    //For each set in the list of sets, check if the set contains the word
    while (tmpSetList != NULL) {
      tmpSet = tmpSetList -> set;
      if (SetContainsWord(tmpSet, tmpWordsList -> word)) {
	AddIntListItem(&wordBelongsToSets, 1);
      } else {
	AddIntListItem(&wordBelongsToSets, 0);
      }
      tmpSetList = tmpSetList -> next;
    }
    AddListToMatrix(&characteristicMatrix, wordBelongsToSets);
    tmpWordsList = tmpWordsList -> next;
    ClearIntList(&wordBelongsToSets);    
  }
  return characteristicMatrix;
}

setList*
CreateSetList()
{

  setList *setList = NULL;
  
  set *set1 = NULL;
  AddItem(&set1,"ball");
  AddItem(&set1,"fork");
  AddItem(&set1,"goal");
  AddItem(&set1,"kick");
  AddSet(&setList,set1);

  set *set2 = NULL;
  AddItem(&set2,"ball");
  AddItem(&set2,"concert");
  AddItem(&set2,"enter");
  AddItem(&set2,"fork");
  AddItem(&set2,"goal");
  AddItem(&set2,"kick");
  AddSet(&setList,set2);
  
  set *set3 = NULL;
  AddItem(&set3,"airplane");
  AddItem(&set3,"ball");
  AddItem(&set3,"concert");
  AddItem(&set3,"goal");
  AddItem(&set3,"junior");
  AddSet(&setList,set3);
  
  set *set4 = NULL;
  AddItem(&set4,"doll");
  AddItem(&set4,"goal");
  AddItem(&set4,"horse");
  AddItem(&set4,"junior");
  AddSet(&setList,set4);

  return setList;
}

int
hash()
{
  int a = (int)rand();
  int b = (int)rand();
  int c = (int)rand();

  int hashValue = ((a * ((a*b*c) >> 4) + b * (a*b*c) + c) & 131071);
  
  return hashValue;
}

intMatrix*
BuildHashMatrix(setList *aSetList, wordsList *aWordsList)
{
  int numHashFunctions = NUM_HASH_FUNCTIONS;
  intMatrix *hashMatrix = NULL;
  intList *hashedWordForSets = NULL;
  wordsList *tmpWordsList = aWordsList;
  
  //For each unique word in the universal set
  while (tmpWordsList != NULL) {

    //For each word, compute the hash value
    int i;
    for (i = 0; i < numHashFunctions; i++) {
      AddIntListItem(&hashedWordForSets, hash());
    }
    AddListToMatrix(&hashMatrix, hashedWordForSets);
    tmpWordsList = tmpWordsList -> next;
    ClearIntList(&hashedWordForSets);
  }

  return hashMatrix;
}

intMatrix*
BuildSignatureMatrix(int numSets)
{
  int numHashFunctions = NUM_HASH_FUNCTIONS;
  intMatrix *signatureMatrix = NULL;

  InitializeSignatureMatrix(&signatureMatrix, numSets, numHashFunctions);

  return signatureMatrix;
}

void
UpdateSignatureMatrix(intMatrix **aSignatureMatrix, intList *characteristicMatrixRow, intList *hashMatrixRow)
{
  int numHashFunctions = NUM_HASH_FUNCTIONS;
  //  intMatrix *tmpSignatureMatrix = *aSignatureMatrix;
  int setOffset = 0;
  intList *tmpHMRow = NULL;
  intList *tmpSMRow = NULL;
  while (characteristicMatrixRow != NULL) {
    if (characteristicMatrixRow -> elementValue) {
      GetMatrixRowReference(aSignatureMatrix, &tmpSMRow, setOffset*numHashFunctions);
      tmpHMRow = hashMatrixRow;
      while ((tmpHMRow != NULL) && (tmpSMRow != NULL)) {
	if ((tmpHMRow -> elementValue) < (tmpSMRow -> elementValue)) {
	  tmpSMRow -> elementValue = tmpHMRow -> elementValue;
	} 
	tmpHMRow = tmpHMRow -> next;
	tmpSMRow = tmpSMRow -> next;
      }
    }
    setOffset++;
    characteristicMatrixRow = characteristicMatrixRow -> next;
  }
}

void
ComputeSignatureMatrix(intMatrix **aSignatureMatrix, intMatrix *aCharacteristicMatrix, intMatrix *aHashMatrix, wordsList *aWordsList, int numSets)
{
  int numHashFunctions = NUM_HASH_FUNCTIONS;
  int wordsOffset = 0;
  intList *tmpCMRow = NULL;
  intList *tmpHMRow = NULL;
  while (aWordsList != NULL) {
    if ((aCharacteristicMatrix != NULL) && (aHashMatrix != NULL)) {
      tmpCMRow = GetMatrixRow(aCharacteristicMatrix, wordsOffset*numSets);
      tmpHMRow = GetMatrixRow(aHashMatrix, wordsOffset*numHashFunctions);
      UpdateSignatureMatrix(aSignatureMatrix, tmpCMRow, tmpHMRow);
    }
    aWordsList = aWordsList -> next;
    wordsOffset++;
  }
}

float
CaltulateSimilarityBetweenSets(intList *outerSMRow, intList *innerSMRow, int numHashFunctions)
{
  int sameMinhashQuantity = 0;
  int i;
  for (i = 0; i < numHashFunctions; i++) {
    if ((outerSMRow != NULL) && (innerSMRow != NULL)) {
      if ((outerSMRow -> elementValue) == (innerSMRow -> elementValue)) {
	sameMinhashQuantity++;
      }
      outerSMRow = outerSMRow -> next;
      innerSMRow = innerSMRow -> next;
    }
  }
  return ((sameMinhashQuantity * 1.0)/(numHashFunctions * 1.0));
}

void
ComputeSimilarities(intMatrix *aSignatureMatrix)
{
  int numHashFunctions = NUM_HASH_FUNCTIONS;
  int outerOffset = 0;
  int innerOffset = 0;

  intList *tmpSMOuter = GetMatrixRow(aSignatureMatrix, outerOffset*numHashFunctions);
  intList *tmpSMInner = GetMatrixRow(aSignatureMatrix, innerOffset*numHashFunctions);

  while (tmpSMOuter != NULL) {
    innerOffset = 0;
    tmpSMInner = GetMatrixRow(aSignatureMatrix, innerOffset*numHashFunctions);
    while (tmpSMInner != NULL) {
      if (outerOffset != innerOffset) {
	float similarityBetweenSets = CaltulateSimilarityBetweenSets(tmpSMOuter, tmpSMInner, numHashFunctions);
	printf("The similarity between S%d and S%d is %f.\n", outerOffset, innerOffset, similarityBetweenSets);
      }
      innerOffset++;
      tmpSMInner = GetMatrixRow(aSignatureMatrix, innerOffset*numHashFunctions);
    }
    outerOffset++;
    tmpSMOuter = GetMatrixRow(aSignatureMatrix, outerOffset*numHashFunctions);
  }
}

int
main (int argc, char *argv[])
{
  srand(time(NULL));
  
  setList *setList = CreateSetList();
  printf("setListSize: %d\n", SetListSize(setList));
  wordsList *wordsList = BuildWordsList(setList);

  intMatrix *characteristicMatrix = BuildCharacteristicMatrix(setList, wordsList);
  printf("charMatrixSize: %d\n", IntMatrixSize(characteristicMatrix));

  intMatrix *hashMatrix = BuildHashMatrix(setList,wordsList);
  printf("intMatrixSize: %d\n", IntMatrixSize(hashMatrix));
  
  intMatrix *signatureMatrix = BuildSignatureMatrix(SetListSize(setList));
  printf("signatureMatrixSize: %d\n", IntMatrixSize(signatureMatrix));

  ComputeSignatureMatrix(&signatureMatrix, characteristicMatrix, hashMatrix, wordsList, SetListSize(setList));
  //PrintIntMatrix(signatureMatrix);

  ComputeSimilarities(signatureMatrix);
  
  ClearIntMatrix(&characteristicMatrix);
  ClearIntMatrix(&hashMatrix);
  ClearIntMatrix(&signatureMatrix);  
  ClearWordsList(&wordsList);
  ClearSetList(&setList);
  
  return 0;
}

