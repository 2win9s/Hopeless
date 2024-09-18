// Stuck on C++17 so I need to implement a simple dynamic extent span 
// also so I can use omp declare target
#pragma once

#ifndef HOPELESS_SPAN
#define HOPELESS_SPAN

#include<iostream>
#include<type_traits>
#include<array>
#include<omp.h>

#include "hopeless_macros_n_meta.hpp"

namespace hopeless{
    template<typename T>
    struct dyn_extent_span;

    template<typename T>     
    auto inline operator <<(std::ostream& stream,
    const dyn_extent_span<T> & span) 
        -> type_<std::ostream&, 
            decltype(std::cout<<std::declval<T>())>         
    {
        constexpr bool is_array_of_obj = std::is_class<T>::value;
        constexpr auto default_sep = (is_array_of_obj) ? ",\n":", ";
        stream << "[";
        for(intptr_t i = 0; i < span.size()-1; ++i){
            stream << span[i] << default_sep;
        }
        if (span.size()){
            stream << span[span.size()-1];
        }
        stream << "]";
        return stream;
    }

    template<typename T>
    struct dyn_extent_span
    {
        // declare iterators here to use for typedef
        struct rand_access_iterator;
        struct const_rand_access_iterator;

        typedef T element_type;
        typedef typename std::remove_cv<T> value_type;
        typedef T& reference;
        typedef const T& const_reference;
        typedef T* pointer;
        typedef const T * const_pointer;
        typedef std::ptrdiff_t difference_type;
        typedef std::ptrdiff_t size_type; 
        typedef rand_access_iterator iterator;
        typedef const_rand_access_iterator const_iterator;
        
        typedef typename std::reverse_iterator<iterator> reverse_iterator;
        typedef typename std::reverse_iterator<const_iterator> const_reverse_iterator;

        constexpr inline reference operator [](const size_type i)const noexcept;
        constexpr inline reference operator ()(const size_type i)const noexcept;

        struct rand_access_iterator  
        {
        public:
            typedef std::random_access_iterator_tag iterator_category;   
            typedef T value_type;
            typedef value_type& reference;
            typedef const value_type& const_reference;
            typedef std::ptrdiff_t difference_type;
            typedef value_type* pointer;	
            typedef const value_type* const_pointer;	
        protected:
            pointer iter;       // the iterator will simply be a pointer
        public:
            constexpr rand_access_iterator()noexcept:iter(nullptr){}
            constexpr rand_access_iterator(pointer ptr)noexcept:iter(ptr){}
            constexpr rand_access_iterator(const_pointer ptr)noexcept:iter(const_cast<pointer>(ptr)){}
            rand_access_iterator(intptr_t ptr)noexcept:iter(reinterpret_cast<pointer>(ptr)){}
            constexpr rand_access_iterator(const rand_access_iterator & it)noexcept:iter(it.iter){}
            rand_access_iterator & operator =(const rand_access_iterator & it)noexcept{iter = it.iter;return *this;}

            inline rand_access_iterator & operator ++()noexcept{++iter;return *this;}
            inline rand_access_iterator  operator ++(int)noexcept{auto temp = *this; ++iter; return temp;}
            inline rand_access_iterator & operator --()noexcept{--iter;return *this;}
            inline rand_access_iterator  operator --(int)noexcept{auto temp = *this; --iter; return temp;}
            inline rand_access_iterator & operator +=(const difference_type n)noexcept{iter+=n;return *this;}
            inline rand_access_iterator & operator -=(const difference_type n)noexcept{iter-=n;return *this;}
            inline rand_access_iterator  operator +(const difference_type n)const noexcept{return rand_access_iterator(iter + n);}
            inline rand_access_iterator  operator -(const difference_type n)const noexcept{return rand_access_iterator(iter - n);}
            
            inline friend rand_access_iterator operator +(const difference_type n, const rand_access_iterator it)noexcept{return rand_access_iterator (it.iter + n);}
            inline friend rand_access_iterator operator -(const difference_type n, const rand_access_iterator it)noexcept{return rand_access_iterator (it.iter - n);}
            
            inline reference operator *()const noexcept{return *iter;}
            inline pointer operator ->()const noexcept{return iter;}
            inline reference operator [](const difference_type n)const noexcept{return iter[n];}
            
            inline pointer ptr()const noexcept{return iter;}

            inline friend bool operator > (const rand_access_iterator lhs, const rand_access_iterator rhs)noexcept{return (lhs.iter>rhs.iter);}
            inline friend bool operator < (const rand_access_iterator  lhs, const rand_access_iterator rhs)noexcept{return (lhs.iter<rhs.iter);}
            inline friend bool operator != (const rand_access_iterator  lhs, const rand_access_iterator rhs)noexcept{return (lhs.iter!=rhs.iter);}
            inline friend bool operator == (const rand_access_iterator  lhs, const rand_access_iterator rhs)noexcept{return (lhs.iter==rhs.iter);}
            inline friend bool operator >= (const rand_access_iterator  lhs, const rand_access_iterator rhs)noexcept{return (lhs.iter>=rhs.iter);}
            inline friend bool operator <= (const rand_access_iterator  lhs, const rand_access_iterator rhs)noexcept{return (lhs.iter<=rhs.iter);}
            inline friend bool operator != (const rand_access_iterator  lhs, const const_rand_access_iterator rhs)noexcept{return (lhs.iter!=rhs.iter);}
            inline friend difference_type operator -(const rand_access_iterator lhs, const rand_access_iterator rhs)noexcept{return (lhs.iter - rhs.iter);}
        };

        // inherit from rand_access_iterator but return const reference instead
        struct const_rand_access_iterator : public rand_access_iterator  
        {
        public:
            typedef std::random_access_iterator_tag iterator_category;   
            typedef T value_type;
            typedef value_type& reference;
            typedef const value_type& const_reference;
            typedef std::ptrdiff_t difference_type;
            typedef value_type* pointer;	
            typedef const value_type* const_pointer;	

            const_rand_access_iterator()noexcept:rand_access_iterator(nullptr){}
            const_rand_access_iterator(pointer ptr)noexcept:rand_access_iterator(ptr){}
            const_rand_access_iterator(const_pointer ptr)noexcept:rand_access_iterator(ptr){}
            const_rand_access_iterator(intptr_t ptr)noexcept:rand_access_iterator(ptr){}
            const_rand_access_iterator(const rand_access_iterator &it)noexcept:rand_access_iterator(it){}
            const_rand_access_iterator(const_rand_access_iterator & it)noexcept:rand_access_iterator(it){}
            
            const_rand_access_iterator& operator=(const rand_access_iterator & it)noexcept{rand_access_iterator::operator=(it);return *this;}

            inline const_reference operator *()const noexcept {return *(this->iter);}
            inline const_pointer operator ->()const noexcept {return this->iter;}
            inline const_reference operator [](const difference_type n)const noexcept {return static_cast<const_reference>(this->iter[n]);}

            inline friend bool operator > (const const_rand_access_iterator  lhs, const const_rand_access_iterator rhs)noexcept{return (lhs.iter>rhs.iter);}
            inline friend bool operator < (const const_rand_access_iterator  lhs, const const_rand_access_iterator  rhs)noexcept{return (lhs.iter<rhs.iter);}
            inline friend bool operator != (const const_rand_access_iterator  lhs, const const_rand_access_iterator rhs)noexcept{return (lhs.iter!=rhs.iter);}
            inline friend bool operator == (const const_rand_access_iterator  lhs, const const_rand_access_iterator rhs)noexcept{return (lhs.iter==rhs.iter);}
            inline friend bool operator >= (const const_rand_access_iterator  lhs, const const_rand_access_iterator rhs)noexcept{return (lhs.iter>=rhs.iter);}
            inline friend bool operator <= (const const_rand_access_iterator  lhs, const const_rand_access_iterator  rhs)noexcept{return (lhs.iter<=rhs.iter);}
            inline friend const_rand_access_iterator operator +(const difference_type n, const const_rand_access_iterator  it)noexcept{return const_rand_access_iterator (it.iter + n);}
            inline friend const_rand_access_iterator operator -(const difference_type n, const const_rand_access_iterator  it)noexcept{return const_rand_access_iterator (it.iter - n);}
            inline friend difference_type operator -(const const_rand_access_iterator lhs, const const_rand_access_iterator rhs)noexcept{return (lhs.iter - rhs.iter);}
        };

        constexpr dyn_extent_span() noexcept;

        template<typename contin_It >       // obviously UB if first is pointing to NULL
        constexpr dyn_extent_span(contin_It first, size_type count)noexcept;

        constexpr dyn_extent_span(pointer first, size_type count)noexcept;

        template<typename U, std::size_t N>
        constexpr dyn_extent_span(U (&array)[N]) noexcept;  

        template<typename U, std::size_t N>
        constexpr dyn_extent_span(std::array<U, N>& array) noexcept;  

        constexpr dyn_extent_span(std::initializer_list<value_type> ilist) noexcept;

        constexpr dyn_extent_span(const dyn_extent_span & other) noexcept = default;

        constexpr inline iterator begin()const noexcept;
        constexpr inline const_iterator cbegin()const noexcept;
        constexpr inline iterator end()const noexcept;
        constexpr inline const_iterator cend()const noexcept;
        
        constexpr inline reverse_iterator rbegin()const noexcept;
        constexpr inline const_reverse_iterator crbegin()const noexcept;
        constexpr inline reverse_iterator rend()const noexcept;
        constexpr inline const_reverse_iterator crend()const noexcept;

        constexpr inline reference front() const noexcept;
        constexpr inline reference back() const noexcept;
        constexpr inline reference at(size_type pos) const noexcept;
        constexpr inline pointer data() const noexcept;
        constexpr inline size_type size() const noexcept;
        constexpr inline size_type size_bytes() const noexcept;
        constexpr inline bool empty()const noexcept;

        constexpr inline void resize(size_type new_size)noexcept;
        // Function for changing the pointer after pointer invalidation
        constexpr inline void change_span_ptr(pointer ptr)noexcept;

    private:
        pointer   data_;
        size_type size_;
    };
    
    template<typename T>
    constexpr inline dyn_extent_span<T>::reference dyn_extent_span<T>::operator [](const size_type i)const noexcept{
        return data_[i];
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::reference dyn_extent_span<T>::operator ()(const size_type i)const noexcept{
        return data_[i];
    }

    template<typename T>
    constexpr dyn_extent_span<T>::dyn_extent_span()noexcept
    :data_(nullptr),
    size_(0){}

    template<typename T>
    template<typename contin_It >
    constexpr dyn_extent_span<T>::dyn_extent_span(contin_It first, size_type count)noexcept
    :data_(const_cast<pointer>(&*first)),
    size_(count){}

    template<typename T>
    constexpr dyn_extent_span<T>::dyn_extent_span(pointer first, size_type count)noexcept
    :data_(first),
    size_(count){}

    template<typename T>
    template<typename U, std::size_t N>
    constexpr dyn_extent_span<T>::dyn_extent_span(U (&array)[N])noexcept
    :data_(const_cast<T*>(&array[0])),
    size_(N){static_assert(std::is_same_v<std::remove_cv_t<T>,std::remove_cv_t<U>>,"Array of wrong type passed to hopeless dyn_extent_span constructor,\n type punning this way is UB in C++");}

    template<typename T>
    template<typename U, std::size_t N>
    constexpr dyn_extent_span<T>::dyn_extent_span(std::array<U, N>& array)noexcept
    :data_(const_cast<T*>(&array[0])),
    size_(N){static_assert(std::is_same_v<std::remove_cv_t<T>,std::remove_cv_t<U>>,"Array of wrong type passed to hopeless dyn_extent_span constructor,\n type punning this way is UB in C++");}

    template<typename T>
    constexpr dyn_extent_span<T>::dyn_extent_span(std::initializer_list<value_type> ilist)noexcept
    :data_(const_cast<T*>(&(*ilist.begin()))),
    size_(ilist.size()){}

    template<typename T>
    constexpr inline dyn_extent_span<T>::iterator dyn_extent_span<T>::begin()const noexcept{
        return iterator(data_);
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::const_iterator dyn_extent_span<T>::cbegin()const noexcept{
        return const_iterator(data_);
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::iterator dyn_extent_span<T>::end()const noexcept{
        return iterator(data_+size_);
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::const_iterator dyn_extent_span<T>::cend()const noexcept{
        return const_iterator(data_+size_);
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::reverse_iterator dyn_extent_span<T>::rbegin()const noexcept{
        return reverse_iterator(iterator(data_));
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::const_reverse_iterator dyn_extent_span<T>::crbegin()const noexcept{
        return const_reverse_iterator(const_iterator(data_));
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::reverse_iterator dyn_extent_span<T>::rend()const noexcept{
        return reverse_iterator(iterator(data_+size_));
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::const_reverse_iterator dyn_extent_span<T>::crend()const noexcept{
        return const_reverse_iterator(const_iterator(data_+size_));
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::reference dyn_extent_span<T>::front()const noexcept{
        return data_[0];
    }
    
    template<typename T>
    constexpr inline dyn_extent_span<T>::reference dyn_extent_span<T>::back()const noexcept{
        return data_[size_-1];
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::reference dyn_extent_span<T>::at(size_type pos)const noexcept{
        if(pos<size_)&&(pos>0){
            return data_[pos];
        }else{
            std::cerr<<"Hopeless dyn_extent_span::at(" <<pos<< ") out of bounds for dyn_extent_span with size_" <<size_<<"\n";
        }
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::pointer dyn_extent_span<T>::data()const noexcept{
        return data_;
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::size_type dyn_extent_span<T>::size()const noexcept{
        return size_;
    }

    template<typename T>
    constexpr inline dyn_extent_span<T>::size_type dyn_extent_span<T>::size_bytes()const noexcept{
        return size_ * sizeof(T);
    }

    template<typename T>
    constexpr inline bool dyn_extent_span<T>::empty()const noexcept{
        return (size_ == 0);
    }

    template<typename T>
    constexpr inline void dyn_extent_span<T>::resize(size_type new_size)noexcept{
        size_=new_size;
    }

    template<typename T>
    constexpr inline void dyn_extent_span<T>::change_span_ptr(pointer ptr)noexcept{
        data_=ptr;
    }
}
#endif