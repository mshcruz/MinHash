#include "intMatrix.h"

intList*
GetMatrixRow(intMatrix *anIntMatrix, int offSet)
{
  intList *tmpIntList = (intList*)anIntMatrix;
  int i;
  for (i = 0; i < offSet; i++) {
    if (tmpIntList != NULL) {
      tmpIntList = tmpIntList -> next;
    }
  }

  return tmpIntList;
}

void
GetMatrixRowReference(intMatrix **anIntMatrix, intList **buffer, int offSet)
{
  (*buffer) = (intList*)*anIntMatrix;
  int i;
  for (i = 0; i < offSet; i++) {
    if ((*buffer) != NULL) {
      (*buffer) = (*buffer) -> next;
    }
  }
}

void
AddListToMatrix(intMatrix **aIntMatrix, intList *anIntList)
{
  intList *tmpIntList = anIntList;
  while (tmpIntList != NULL) {
    AddIntMatrixElement(aIntMatrix, tmpIntList -> elementValue);
    tmpIntList = tmpIntList -> next;
  }
}

void
AddIntMatrixElement (intMatrix **aIntMatrix, int elementValue)
{
  intMatrix *tmpIntMatrix = *aIntMatrix;

  if (tmpIntMatrix != NULL) {
    while (tmpIntMatrix -> next != NULL) {
      tmpIntMatrix = tmpIntMatrix -> next;
    }
    tmpIntMatrix -> next = malloc(sizeof(intMatrix));
    tmpIntMatrix = tmpIntMatrix -> next;
    tmpIntMatrix -> next = NULL;
    tmpIntMatrix -> elementValue = elementValue;
  } else {
    (*aIntMatrix) = malloc(sizeof(intMatrix));
    (*aIntMatrix) -> next = NULL;
    (*aIntMatrix) -> elementValue = elementValue;
  }
}    

void
PrintIntMatrix (intMatrix *aIntMatrix)
{
  if (aIntMatrix == NULL)
    printf ("The Int Matrix is empty!\n");
  else
    while (aIntMatrix != NULL) {
      printf ("%d\t", aIntMatrix -> elementValue);
      aIntMatrix = aIntMatrix -> next;
    }
  printf ("\n");
}

int
IntMatrixSize (intMatrix *aIntMatrix)
{
  int aIntMatrixSize = 0;
  while (aIntMatrix != NULL) {
    aIntMatrixSize++;
    aIntMatrix = aIntMatrix -> next;
  }
  return aIntMatrixSize;
}

void
ClearIntMatrix (intMatrix **aIntMatrix)
{
  while ((*aIntMatrix) != NULL) {
    intMatrix *tmpIntMatrix = *aIntMatrix;
    (*aIntMatrix) = (*aIntMatrix) -> next;
    free(tmpIntMatrix);
  }
}
