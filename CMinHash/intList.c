#include "intList.h"

void
AddIntListItem (intList **anIntList, int elementValue)
{
  intList *tmpIntList = *anIntList;
  if (tmpIntList != NULL) {
    while (tmpIntList -> next != NULL)
      tmpIntList = tmpIntList -> next;
    tmpIntList -> next = malloc(sizeof(intList));
    tmpIntList = tmpIntList -> next;
    tmpIntList -> next = NULL;
    tmpIntList -> elementValue = elementValue;
  } else {
    (*anIntList) = malloc(sizeof(intList));
    (*anIntList) -> next = NULL;
    (*anIntList) -> elementValue = elementValue;
  }
}    

void
PrintIntList (intList *anIntList)
{
  if (anIntList == NULL)
    printf ("Int List is empty!\n");
  else
    while (anIntList != NULL) {
      printf ("%d\t", anIntList -> elementValue);
      anIntList = anIntList -> next;
    }
  printf ("\n");
}

int
IntListSize (intList *anIntList)
{
  int intListSize = 0;
  while (anIntList != NULL) {
    intListSize++;
    anIntList = anIntList -> next;
  }
  return intListSize;
}

void
ClearIntList (intList **anIntList)
{
  while ((*anIntList) != NULL) {
    intList *tmp = *anIntList;
    (*anIntList) = (*anIntList) -> next;
    free (tmp);
  }
}
