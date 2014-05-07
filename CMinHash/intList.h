#include <stdio.h>
#include <stdlib.h>

#ifndef INTLIST_H
#define INTLIST_H
typedef struct intList intList;
struct intList{
  int elementValue;
  intList *next;
};
#endif
void AddIntListItem (intList**, int);
void PrintIntList (intList*);
int IntListSize (intList*);
void ClearIntList (intList**);
