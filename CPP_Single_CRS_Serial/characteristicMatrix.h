#include <vector>
#include <string>
#include <map>

typedef struct crsMatrix crsMatrix;
struct crsMatrix{
  //  vector<int> val;
  std::vector<int> col_ind;
  std::vector<int> row_ptr;
};

crsMatrix* buildCharacteristicMatrix(std::multimap<std::string, int>);
void printCharacteristicMatrixVectors (crsMatrix*);
int crsMatrixSize (crsMatrix*);
