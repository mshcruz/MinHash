#include <stdio.h>
#include <stdlib.h>
#include "set.h"

typedef struct setList setList;
struct setList{
  set *set;
  setList *next;
};

void AddSet (setList**, set*);
int SetListSize (setList*);
void ClearSetList (setList**);
void PrintSetList (setList*);
