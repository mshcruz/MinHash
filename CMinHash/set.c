#include "set.h"

void
AddItem (set **aSet, const char *elementValue)
{
  set *tmp = *aSet;
  if (tmp != NULL) {
    while (tmp -> next != NULL)
      tmp = tmp -> next;
    tmp -> next = malloc(sizeof(set));
    tmp = tmp -> next;
    tmp -> next = NULL;
    tmp -> elementValue = elementValue;
  } else {
    (*aSet) = malloc(sizeof(set));
    (*aSet) -> next = NULL;
    (*aSet) -> elementValue = elementValue;
  }
}    

int
SetContainsWord (set *aSet, const char *elementValue)
{
  //Sequential Search
  while (aSet != NULL) {
    if (strcmp(aSet -> elementValue, elementValue) == 0) {
      return 1;
    }
    aSet = aSet -> next;
  }
  return 0;
}

void
PrintSet (set *aSet)
{
  if (aSet == NULL)
    printf ("Set is empty!\n");
  else
    while (aSet != NULL) {
      printf ("%s\t", aSet -> elementValue);
      aSet = aSet -> next;
    }
  printf ("\n");
}

int
SetSize (set *aSet)
{
  int setSize = 0;
  while (aSet != NULL) {
    setSize++;
    aSet = aSet -> next;
  }
  return setSize;
}

void
ClearSet (set **aSet)
{
  while ((*aSet) != NULL) {
    set *tmp = *aSet;
    (*aSet) = (*aSet) -> next;
    free (tmp);
  }
}

