ICS-E4020 Programming Parallel Computers
========================================

Material for the weekly exercises.

See the course web page for details:
https://users.ics.aalto.fi/suomela/ppc-2015/


Files to edit
-------------

Weekly reports:

    report/week*.pdf

Tasks:

    mf*/mf.cc


Quick start
-----------

Let us use task MF1 as an example:

    cd mf1
    make test

The test should fail, as we have not implemented the median filter
subroutine yet. Now open the file `mf.cc` in a text editor and
fill in the missing details. Then compile it:

    make

Run the test suite:

    make test

And do some benchmarking to see that it performs well:

    make benchmark

In the benchmarks, the last column is the running time in seconds.
For more thorough benchmarks, try:

    make benchmark2


Environment
-----------

We will assume the following:

 - Operating system: Linux or Mac OS X.

 - Compiler: GCC version 4.9 or 4.8, somewhere in the path,
   with the name `g++-4.9`, `g++-4.8`, or `g++`.

 - Libraries: libpng installed in a location where GCC can find it.


### Classroom computers

Everything should work directly in the classroom computers (Maari-A).


### Your own computers

To run this on your own OS X computer, try the following:

 - Install Homebrew: http://brew.sh/

 - Run: `brew install gcc libpng`

To run this on your own Ubuntu 14.04 Linux computer, try the
following:

 - Run: `sudo apt-get install g++-4.8 libpng12-dev`
