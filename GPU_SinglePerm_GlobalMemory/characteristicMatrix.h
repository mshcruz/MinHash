#include <vector>
#include <string>
#include <unordered_map>
#include "kernel.h"

using namespace std;
/*
typedef struct ccsMatrix ccsMatrix;
struct ccsMatrix{
  //vector<int> val;
  vector<int> row_ind;
  vector<int> col_ptr;
};
*/
ccsMatrix* buildCharacteristicMatrix(unordered_multimap<int,int>);
void printCharacteristicMatrixVectors (ccsMatrix*);
int ccsMatrixSize (ccsMatrix*);
