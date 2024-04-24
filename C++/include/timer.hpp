#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>

template <typename Clock = std::chrono::steady_clock,
          typename Duration = typename Clock::duration>
/**
 *  * class used in main_benchmark.cc in order to simply measure an initial and final times of an operation
 *   */
class timer {
  using time_point = std::chrono::time_point<Clock, Duration>;
  /**
 *    * initial time
 *       */
  time_point t0;

 public:
  /**
 *    * set initial time
 *       */
  void start() { t0 = Clock::now(); }
  /**
 *    * set final time, diirectly returing the difference
 *       */  
  double stop() {
    time_point t1 = Clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(t1 -t0).count(); //seconds
  }
};

/*usage:
 *
 *timer<> t;
 *t.start();
 *auto appo=t.stop();
 */
