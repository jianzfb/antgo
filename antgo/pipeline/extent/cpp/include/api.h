#ifndef MOBULA_INCLUDE_API_H_
#define MOBULA_INCLUDE_API_H_

#ifdef _WIN32
#define MOBULA_DLL __declspec(dllexport)
#else
#define MOBULA_DLL
#endif

#endif  // MOBULA_INCLUDE_API_H_
