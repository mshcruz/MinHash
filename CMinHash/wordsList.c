#include "wordsList.h"

void
AddWord (wordsList **aWordsList, const char* aWord)
{
  wordsList *tmpWordsList = *aWordsList;
  if (tmpWordsList != NULL) {
    if (!ContainsWord(*aWordsList, aWord)) {
      while (tmpWordsList -> next != NULL)
	tmpWordsList = tmpWordsList -> next;
      tmpWordsList -> next = malloc(sizeof(wordsList));
      tmpWordsList = tmpWordsList -> next;
      tmpWordsList -> next = NULL;
      tmpWordsList -> word = aWord;
    }
  } else {
    (*aWordsList) = malloc(sizeof(wordsList));
    (*aWordsList) -> next = NULL;
    (*aWordsList) -> word = aWord;
  }
}    

int
ContainsWord (wordsList *aWordsList, const char* aWord)
{
  //Sequential search
  while (aWordsList != NULL) {
    if (strcmp(aWordsList -> word, aWord) == 0) {
      return 1;
    }
    aWordsList = aWordsList -> next;
  }
  return 0;
}

void
PrintWordsList (wordsList *aWordsList) {
  if (aWordsList == NULL)
    printf ("The word list is empty!\n");
  else
    while (aWordsList != NULL) {
      printf ("%s\t", aWordsList -> word);
      aWordsList = aWordsList -> next;
    }
  printf ("\n");
}

int
WordsListSize (wordsList *aWordsList)
{
  int wordsListSize = 0;
  while (aWordsList != NULL) {
    wordsListSize++;
    aWordsList = aWordsList -> next;
  }
  return wordsListSize;
}

void
ClearWordsList (wordsList **aWordsList)
{
  while ((*aWordsList) != NULL) {
    wordsList *tmpWordsList = *aWordsList;
    (*aWordsList) = (*aWordsList) -> next;
    free (tmpWordsList);
  }
}
