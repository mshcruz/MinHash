#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct set set;
struct set{
  const char *elementValue;
  set *next;
};

void AddItem (set**, const char*);
int SetContainsWord(set*, const char*);
void PrintSet (set*);
int SetSize (set*);
void ClearSet (set**);
