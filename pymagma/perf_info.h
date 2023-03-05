#ifndef PERF_INFO_H
#define PERF_INFO_H

#include <chrono>

// Performance information from the solvers
// Using a struct because it seems likely that there will
// more members added later (data transfer, breaking down the solvers into steps)

struct PerfInfo
{
    double elapsed;
};

// Use destructor to automically record the elapsed time
class RecordElapsed
{
public:
    using timer = std::chrono::high_resolution_clock;

    RecordElapsed(PerfInfo &perf) : perf_(perf)
    {
        t0 = timer::now();
    }

    ~RecordElapsed()
    {
        timer::time_point t1 = timer::now();
        perf_.elapsed = std::chrono::duration<double>(t1 - t0).count();
    }

private:
    PerfInfo& perf_;
    timer::time_point t0;
};

#endif
