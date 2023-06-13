#ifndef ANTGO_INCLUDE_API_H_
#define ANTGO_INCLUDE_API_H_

#ifdef _WIN32
#define ANTGO_API __declspec(dllexport)
#else
#define ANTGO_API
#endif

#endif  // MOBULA_INCLUDE_API_H_
