#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

typedef struct crsMatrix crsMatrix;
struct crsMatrix{
  //  vector<int> val;
  vector<int> col_ind;
  vector<int> row_ptr;
};

crsMatrix* buildCharacteristicMatrix(unordered_multimap<string, int>);
void printCharacteristicMatrixVectors (crsMatrix*);
int crsMatrixSize (crsMatrix*);
