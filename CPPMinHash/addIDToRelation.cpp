// -*- coding: utf-8 -*-
//

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>

using namespace std;

int
main(int argc, char *argv[])
{

  ifstream relation ("newRelationS.data");
  ofstream cleanRelation ("newRelationS_ID.data");
  string line = "";
  
  if (relation.is_open() && cleanRelation.is_open()){
    int count = 0;
    while (getline(relation,line)){
      cleanRelation << "S" << count << "\t" << line << "\n";
      count++;
    }    
    relation.close();
  } else {
    cout << "Error opening file.";
  }


  return 0;
}

