#pragma once

#ifndef HOPELESS_MACROS_N_META
#define HOPELESS_MACROS_N_META

#pragma once

#include <type_traits>
#include <exception>

// define if using openmp offloading
#define HOPELESS_TARGET_OMP_DEV

//define if you want to map changes to device after calls to functions such as insert and push_back (emplace_back is excluded)
//#define HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE              // also applies to r2darray

// control how capcity of dynarray grows
#ifndef HOPELESS_DYNARRAY_CAPACITY_GROWTH_RATE
    #define HOPELESS_DYNARRAY_CAPACITY_GROWTH_RATE 1.61803400516510009765625f  // this is the golden ratio, is it better than 2? I don't know
#endif

// the device number of the device to offload to
#ifdef HOPELESS_TARGET_OMP_DEV
    #define HOPELESS_DEFAULT_OMP_OFFLOAD_DEV 0
#endif

namespace hopeless{

    // a bit of tweaking of std::void_t to use with defered decltype for SFINAE
    template <typename T,typename... Enable> using type_ = T;
    /*  
        example usage for SFINAE with pseudo code:
        template <typename T>
        auto some__func() -> type_t<desired_return_type,decltype(expression_with_T)>
        {
            // stuff that hinges on expression_with_T which might be invlaid for some T
        }


        also can be used in arguement type for same purpose
    */

    inline void cout_exception(std::exception_ptr exception)noexcept{
        try{
            if (exception){std::rethrow_exception(exception);}
        }
        catch(const std::exception& e){std::cerr << e.what() << '\n';}
        catch(...){
            std::cerr << "Unknown failure, possibly custom exception or memory corruption issues?" << "\n";
        }
    }
}
#endif

