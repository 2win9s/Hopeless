# Hopeless

A small programming project where I implement a resizeable runtime array (hopeless::dynarray) and a ragged array(hopeless::r2darray) in C++ which is meant to be used with openmp without explicit mapping
after every target region. I had encountered issues with using containers such as std::vector when using the map directive so I decided to implment these.
Index using () in target regions to specify that you are accessing the device array. The function names are mostly self explanatory but may add documentation in the future (I doubt anyone else will use this)
