#ifndef PERF_INFO_H
#define PERF_INFO_H

#include <chrono>

// Performance information from the solvers
// Using a struct because it seems likely that there will
// more members added later (data transfer, breaking down the solvers into steps)

struct PerfInfo
{
    double elapsed;
    double comp[6]; // Substeps in the solvers. Meaning depends on particular solver.

    double get_comp(int idx) { return comp[idx]; }
};

// Use destructor to automically record the elapsed time
//  Record elapsed time (idx =-1) or a component (idx >= 0)
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

class RecordComp
{
public:
    using timer = std::chrono::high_resolution_clock;

    RecordComp(PerfInfo &perf, int idx) : perf_(perf), idx_(idx)
    {
        t0 = timer::now();
    }

    void stop() {
        timer::time_point t1 = timer::now();
        perf_.comp[idx_] = std::chrono::duration<double>(t1 - t0).count();
    }
private:
    PerfInfo& perf_;
    int idx_;
    timer::time_point t0;
};

#endif
