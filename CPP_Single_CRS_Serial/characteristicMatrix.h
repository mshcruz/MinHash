#include <vector>
#include <string>
#include <unordered_map>

typedef struct crsMatrix crsMatrix;
struct crsMatrix{
  //  vector<int> val;
  std::vector<int> col_ind;
  std::vector<int> row_ptr;
};

crsMatrix* buildCharacteristicMatrix(std::unordered_multimap<std::string, int>);
void printCharacteristicMatrixVectors (crsMatrix*);
int crsMatrixSize (crsMatrix*);
