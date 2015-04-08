#ifndef TIMER_H
#define TIMER_H

#include <iomanip>
#include <iostream>
#include <sys/time.h>

class Timer {
public:
    Timer() : start(get_time())
    {}

    ~Timer() {
        double now = get_time();
        std::cout << std::fixed << std::setprecision(3) << (now - start) << std::flush;
        std::cout.copyfmt(std::ios(NULL));
    }

private:
    static double get_time() {
        struct timeval tm;
        gettimeofday(&tm, NULL);
        return static_cast<double>(tm.tv_sec) + static_cast<double>(tm.tv_usec) / 1E6;
    }

    double start;
};

#endif
