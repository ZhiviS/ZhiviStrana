// Wildcard.cpp

#include "Wildcard.h"

using namespace std;


bool Wildcard::match(const char *str, const char *pattern) {

  const char *s;
  const char *p;
  bool star = false;

loopStart:
  for (s = str, p = pattern; *s; ++s, ++p) {

    switch (*p) {
    case '?':
      if (*s == '.') goto starCheck;
      break;

    case '*':
      star = true;
      str = s, pattern = p;
      if (!*++pattern) return true;
      goto loopStart;

    default:     
        if (*s != *p)
          goto starCheck;     
      break;
    } /* endswitch */

  } /* endfor */

  if (*p == '*') ++p;
  return (!*p);

starCheck:
  if (!star) return false;
  str++;
  goto loopStart;

}
