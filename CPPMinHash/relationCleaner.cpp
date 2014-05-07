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

  ifstream relation ("xhamster.csv");
  ofstream cleanRelation ("cleanBigRelationR.data");
  string line = "";
  
  if (relation.is_open() && cleanRelation.is_open()){
    while (getline(relation,line)){
      string tmp;
      istringstream iss(line);
      getline(iss,tmp,',');
      getline(iss,tmp,',');
      //      if (flagR) {
	getline(iss,tmp,',');      
	//      }
      cleanRelation << tmp << "\n";
    }    
    relation.close();
  } else {
    cout << "Error opening file.";
  }


  return 0;
}

