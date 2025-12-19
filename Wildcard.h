// WILDCARD.H

#ifndef WILDCARDH
#define WILDCARDH

#include <string>

class Wildcard {

public:
  /**
  * Checks whether a string matches a given wildcard pattern.
  * Possible patterns allow to match single characters ('?') or any count of
  * characters ('*')
  */
  static bool match(const char *str, const char *pattern);

};

#endif // WILDCARDH
