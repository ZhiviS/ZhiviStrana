//   GPU/GPUWildcard.h

// ---------------------------------------------------------------------------------
// Wildcard matcher
// ---------------------------------------------------------------------------------

__device__ __noinline__ bool _Match(const char *str, const char *pattern) {

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
      //if (mapCaseTable[*s] != mapCaseTable[*p])
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
