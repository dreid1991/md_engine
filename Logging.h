#pragma once
#ifndef LOGGING_H
#define LOGGING_H

#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct ReturnException : std::exception
{
    char const* what() throw() { return "Stopping program."; }
};

struct AssertFailedException : std::exception
{
    char const* what() throw() { return "Assert failed."; }
};

// Strip path from filename
#define __FILENAME__ std::strrchr(__FILE__, '/') ? \
                     std::strrchr(__FILE__, '/') + 1 : \
                     __FILE__

// Write debug output only if DEBUG is defined
#ifdef DEBUG
    #define DBG 1
#else
    #define DBG 0
#endif

#define Debug(fmt, ...) \
    do { if (DBG) fprintf(stdout, "DEBUG: " fmt "(in %s:%d:%s)\n", \
                          ##__VA_ARGS__, \
                          __FILENAME__, __LINE__, __func__); \
    } while(false)

#define Message(fmt, ...) \
    do { fprintf(stdout, fmt, ##__VA_ARGS__); } while (false)

#define Warning(fmt, ...) \
    do { fprintf(stderr, "WARNING: " fmt " (in %s:%d)\n", \
                 ##__VA_ARGS__, \
                 __FILENAME__, __LINE__); \
    } while(false)

#define Error(fmt, ...) \
    do { fprintf(stderr, "ERROR: " fmt " (in %s:%d)\n", \
                 ##__VA_ARGS__, \
                 __FILENAME__, __LINE__); \
         throw ReturnException(); \
    } while(false)

#define Critical(exception, fmt, ...) \
    do { fprintf(stderr, "ERROR: " fmt " (in %s:%d)\n", \
                 ##__VA_ARGS__, \
                 __FILENAME__, __LINE__); \
         throw exception; \
    } while(false)

#define Fatal(exitCode, fmt, ...) \
    do { fprintf(stderr, "FATAL: " fmt " (in %s:%d)\n", \
                 ##__VA_ARGS__, \
                 __FILENAME__, __LINE__); \
         std::exit(1); \
    } while(false)

#define Assume(test, fmt, ...) \
    do { if (!(test)) { fprintf(stderr, "WARNING: In %s(): Assume " \
                                        #test " failed: " \
                                        fmt " (%s:%d)\n", \
                                __func__, ##__VA_ARGS__, \
                                __FILENAME__, __LINE__); } \
    } while (false)

#define Assert(test, fmt, ...) \
    do { if (!(test)) { fprintf(stderr, "ERROR: In %s(): Assert " \
                                        #test " failed: " \
                                        fmt " (%s:%d)\n", \
                                __func__, ##__VA_ARGS__, \
                                __FILENAME__, __LINE__); \
                        throw AssertFailedException(); } \
    } while (false)

#endif
