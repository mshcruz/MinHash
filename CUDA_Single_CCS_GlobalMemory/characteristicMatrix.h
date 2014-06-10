#include <vector>
#include <map>
#include <cstddef>
#include "kernel.h"

/*
typedef struct ccsMatrix ccsMatrix;
struct ccsMatrix{
  std::vector<int> row_ind;
  std::vector<int> col_ptr;
};
*/

ccsMatrix* buildCharacteristicMatrix(std::multimap<int,int>);
void printCharacteristicMatrixVectors (ccsMatrix*);
int ccsMatrixSize (ccsMatrix*);
