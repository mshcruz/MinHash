#include <vector>
#include <string>
#include <map>
#include "kernel.h"

//typedef struct crsMatrix crsMatrix;
//struct crsMatrix{
//  std::vector<int> col_ind;
//  std::vector<int> row_ptr;
//};

crsMatrix* buildCharacteristicMatrix(std::multimap<std::string, int>);
void printCharacteristicMatrixVectors (crsMatrix*);
int crsMatrixSize (crsMatrix*);
