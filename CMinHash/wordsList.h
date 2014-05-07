#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct wordsList wordsList;
struct wordsList{
  const char *word;
  wordsList *next;
};

void AddWord (wordsList**, const char*);
int ContainsWord (wordsList*, const char*);
void PrintWordsList (wordsList*);
int WordsListSize (wordsList*);
void ClearWordsList (wordsList**);
