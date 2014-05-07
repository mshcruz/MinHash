#include "setList.h"

void
AddSet (setList **aSetList, set *aSet)
{
  setList *tmpSetList = *aSetList;
  if (tmpSetList != NULL) {
    while (tmpSetList -> next != NULL)
      tmpSetList = tmpSetList -> next;
    tmpSetList -> next = malloc(sizeof(setList));
    tmpSetList = tmpSetList -> next;
    tmpSetList -> next = NULL;
    tmpSetList -> set = aSet;
  } else {
    (*aSetList) = malloc(sizeof(setList));
    (*aSetList) -> next = NULL;
    (*aSetList) -> set = aSet;
  }
}    

int
SetListSize (setList *aSetList)
{
  int setListSize = 0;
  while (aSetList != NULL) {
    setListSize++;
    aSetList = aSetList -> next;
  }
  return setListSize;
}

void
PrintSetList (setList *aSetList)
{
  setList *tmpSetList = aSetList;
  while (tmpSetList != NULL) {
    set *tmpSet = tmpSetList -> set;
    while (tmpSet != NULL) {
      printf("%s\t", tmpSet -> elementValue);
      tmpSet = tmpSet -> next;
    }
    printf("\n");
    tmpSetList = tmpSetList -> next;
  }
}

void
ClearSetList (setList **aSetList)
{
  while ((*aSetList) != NULL) {
    setList *temp = *aSetList;
    ClearSet(&(temp -> set));
    (*aSetList) = (*aSetList) -> next;
    free (temp);
  }
}
