#ifndef HOPELESS_ALLOCATOR
#define HOPELESS_ALLOCATOR



#pragma once

#include <iostream>
#include <stdlib.h>
#include <cstddef>
#include <limits>

namespace hopeless
{

    // bare-bones so relys on std::allocator_traits
    template <typename T>
    struct allocator {
        static_assert(!(bool)(sizeof(T)%alignof(T)), "type should be aligned");
    public:
        typedef T value_type;
        typedef T& reference;
        typedef const T& const_reference;
        typedef T* pointer;
        typedef const T * const_pointer;
        typedef std::ptrdiff_t difference_type;
        typedef std::ptrdiff_t size_type;                       // a signed type for size is so much easier to work with arithmetic wise

        pointer allocate (size_type n) noexcept;
        //pointer reallocate (pointer ptr, size_type n) noexcept;  //hmmm seems tricky to do
        void deallocate(pointer ptr, size_type n) noexcept;
        void validate_max(size_type n, size_type max_size) noexcept;
        constexpr size_type max_size() noexcept;
        constexpr friend void swap(allocator & a, allocator & b){
            using std::swap;
        }

        // these are empty because this allocator has no state
        allocator()noexcept{};
        template <typename U>
        allocator(allocator<U> other)noexcept{}
    };
    // as this allocator is just a wrapper around malloc and free with noexcept == should always be true and != false
    template <typename T, typename U>
    bool operator==(const allocator<T>& lhs, const allocator<U>& rhs){  // should the variable names be omitted in arguments if unused?
        return true;
    }

    template <typename T, typename U>
    bool operator!=(const allocator<T>& lhs, const allocator<U>& rhs)
    {
        return false;
    }

    template<typename T>
    allocator<T>::pointer allocator<T>::allocate(size_type n) noexcept{
        if (n == 0) {return nullptr;}
        validate_max(n,max_size());
        using malloc_ptr_noexcept = void* (*)(size_t) noexcept;
        malloc_ptr_noexcept no_throw_call_malloc = reinterpret_cast<malloc_ptr_noexcept>(malloc);
        pointer return_ptr = (pointer)no_throw_call_malloc(sizeof(T) * n);
        if ((bool)(return_ptr)){
            return return_ptr;
        }
        else{
            std::cerr<<"ERROR hopeless::allocator error, call to malloc failed"<<std::endl;
            std::terminate();
        }
    }
    
    // simple wrapper for free
    template<typename T>
    void allocator<T>::deallocate(pointer ptr, size_type n) noexcept{
        using free_ptr_noexcept = void (*)(void *) noexcept;
        free_ptr_noexcept no_throw_call_free = reinterpret_cast<free_ptr_noexcept>(free);
        no_throw_call_free(ptr);
    }

    template<typename T>
    void allocator<T>::validate_max(size_type n, size_type max_size) noexcept{
        if(n > max_size){
            std::cerr<<"ERROR hopeless::allocator error, size of allocation requested is greater than max size"<<std::endl;
            std::terminate();
        }
    }
    template<typename T>
    constexpr allocator<T>::size_type allocator<T>::max_size() noexcept{
        return (std::numeric_limits<allocator<T>::size_type>::max()/sizeof(T))   -1;
    }
}

#endif