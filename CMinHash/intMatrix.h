#include <stdio.h>
#include <stdlib.h>
#include "intList.h"

typedef struct intMatrix intMatrix;
struct intMatrix{
  int elementValue;
  intMatrix *next;
};

void AddListToMatrix (intMatrix**, intList*);
void AddIntMatrixElement (intMatrix**, int);
void PrintIntMatrix (intMatrix*);
int IntMatrixSize (intMatrix*);
void ClearIntMatrix (intMatrix**);
intList* GetMatrixRow (intMatrix*, int);
void GetMatrixRowReference (intMatrix**, intList**, int);
