// implementation of a resizable dynamic array that can be accessed and written to on an openmp offload device through operator ()
// NOTE! iterators aren't meant for the device and won't work there
// designed to work only when offloading to a single device
#pragma once

#ifndef HOPELESS_DYNARRAY
#define HOPELESS_DYNARRAY

#include<iostream>
#include<stdexcept>
#include<cstddef>
#include<type_traits>
#include<memory>
#include<initializer_list>
#include<utility>
#include<limits>
#include<memory_resource>
#include<iterator>
#include<omp.h>

#include "allocator.hpp"
#include "hopeless_macros_n_meta.hpp"
namespace hopeless
{

#ifdef HOPELESS_TARGET_OMP_DEV
    //#pragma omp requires unified_address 

    template<typename T, typename Allocator = hopeless::allocator<T>,
                        int dev_no =HOPELESS_DEFAULT_OMP_OFFLOAD_DEV>
    struct  dynarray;

    template<typename T,typename Allocator,int dev_no>     
    auto inline operator <<(std::ostream& stream,
        const dynarray<T,Allocator,dev_no> & dynamic_array) 
            -> type_<std::ostream&, 
            decltype(std::cout<<std::declval<T>())>;  

    template<typename T,typename Allocator,int dev_no>
    void swap(dynarray<T,Allocator,dev_no>& array1, dynarray<T,Allocator,dev_no>& array2)noexcept;

    template<typename T, typename Allocator,int dev_no>
    struct  dynarray
    {
    public:
        // declare iterators here to use for typedef
        struct rand_access_iterator;
        struct const_rand_access_iterator;

        typedef T value_type;
        typedef Allocator allocator_type;
        typedef std::ptrdiff_t difference_type;
        typedef std::ptrdiff_t size_type;       //I'm sick of the unsigned to signed conversion warning with arithmetic nonsense I'm on a 64bit system when are you going to use an array so big the signed one would fail?
        typedef value_type& reference;
        typedef const value_type& const_reference;
        typedef typename std::allocator_traits<allocator_type>::pointer pointer;	
        typedef typename std::allocator_traits<allocator_type>::const_pointer const_pointer;	
        typedef rand_access_iterator iterator;
        typedef const_rand_access_iterator const_iterator;

        typedef typename std::reverse_iterator<iterator> reverse_iterator;
        typedef typename std::reverse_iterator<const_iterator> const_reverse_iterator;


        static_assert(std::is_trivially_copyable_v<T>, "type should be trivially copyable");
        static_assert(std::is_copy_constructible_v<T>, "type should be trivially copy constructible");
        static_assert(std::is_copy_assignable_v<T>, "type should be trivially copy assignable");
        
        constexpr inline reference operator [](const size_type i)const noexcept;
        constexpr inline reference operator [](const size_type i)noexcept;
        #pragma omp declare target device_type(nohost)
        constexpr inline reference operator ()(const size_type i)const noexcept;
        constexpr inline reference operator ()(const size_type i)noexcept;
        #pragma omp end declare target
        // attempting a random access iterator to try and be compatible with range based for
        struct rand_access_iterator  
        {
        public:
            typedef std::random_access_iterator_tag iterator_category;   
            typedef T value_type;
            typedef value_type& reference;
            typedef const value_type& const_reference;
            typedef typename allocator_type::difference_type difference_type;
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
            typedef typename allocator_type::difference_type difference_type;
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

        // for constructing

        void swap(dynarray<T,Allocator,dev_no>& rhs)noexcept;

        constexpr dynarray()noexcept(noexcept(Allocator()));
        constexpr explicit dynarray(const Allocator& alloc)noexcept;
        dynarray(size_type count, const T& value, const Allocator& alloc = Allocator())noexcept;
        explicit dynarray(size_type count,const Allocator& alloc = Allocator())noexcept;
        
        dynarray( const dynarray& other )noexcept(noexcept(Allocator()));
        dynarray( const dynarray& other, const Allocator& alloc )noexcept;
        dynarray( dynarray&& other)noexcept;
        dynarray( dynarray&& other, const Allocator& alloc)noexcept;
        dynarray( std::initializer_list<T> init,const Allocator& alloc = Allocator())noexcept;
    	template< typename InputIt >
        dynarray(
            type_<InputIt,
                decltype(std::declval<InputIt>().operator->(),
                std::declval<InputIt>().operator*())> first,
            InputIt last,
            const Allocator& alloc = Allocator())noexcept;
        ~dynarray()noexcept;

        template<typename Container,typename ...Last_resort>
        dynarray(const type_<Container,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())> &array,
                const Allocator & alloc = Allocator())noexcept;
        
        template<int dev_no2>
        dynarray& operator =(const dynarray<T,Allocator,dev_no2> other);

        dynarray& operator =(const dynarray & other)noexcept;

        dynarray& operator =(dynarray && other)noexcept;
    
    private:
        // Allocates capacity * sizeof(T) memory + calls construct elements also calls create_dev_buffer()
        template<typename... Args>
        inline void create_dynarr(Args && ...args)noexcept;
        
        // the way the elements are constructed depends on what you pass to it
        inline void construct_elements(const T & value)noexcept;
        
        inline void construct_elements()noexcept;
        
        inline void construct_elements(const size_type begin, 
            const size_type end)noexcept;

        inline void construct_elements(const size_type begin, 
            const size_type end,const_reference value)noexcept;

        // initialise from a ranged based for compatible container
        template<typename Container>
        inline auto construct_elements(const Container & container)noexcept 
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size())>;
        
        template<typename Container>
        inline auto construct_elements(Container && container)noexcept 
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())>;

        template<typename Container>
        inline auto construct_elements(const size_type begin,
            const size_type end,
            const Container & container)noexcept 
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size())>;
        
        // initialise using iterators pointing to a container       
        template< typename InputIt >
        inline auto construct_elements(InputIt first, InputIt last)noexcept 
            -> type_<void,
                decltype(*std::declval<InputIt>()),
                decltype(++std::declval<InputIt>()),
                decltype(first != last)
                >;  // check if it acts like an iterator

        // all constructed elements up to size_
        inline void destroy_elements()noexcept;                             // this one doesn't alter the member variable size_
        inline void destroy_elements(const size_type new_size,const size_type old_size)noexcept;    // this one does
        inline void destroy_dealloc()noexcept;                              // frees up resources

        // ensure size_ and capacity_ are up to date
        inline void create_dev_buffer()noexcept;
        inline void dev_buffer_reinit()noexcept;


    // member functions 
    public:
        // will have many of the features std::vector has
        inline void assign(size_type count, const_reference value)noexcept;       
        template<typename InputIt >
        inline void assign( 
            type_<InputIt,
                decltype(std::declval<InputIt>().operator->(),
                std::declval<InputIt>().operator*())> first,
            InputIt last )noexcept;
        inline void assign( std::initializer_list<T> ilist )noexcept;
        constexpr inline allocator_type get_allocator()const noexcept;
        constexpr inline reference at( size_type pos );
        constexpr inline const_reference at( size_type pos )const;
        constexpr inline reference front();
        constexpr inline const_reference front()const;
        constexpr inline reference back();
        constexpr inline const_reference back()const;
        constexpr inline T* data()noexcept;  //get underlying buffer
        constexpr inline const T* data()const noexcept;  //get underlying buffer
        constexpr inline T* data_dev()noexcept; //get buffer on device
        constexpr inline iterator begin() noexcept;
        constexpr inline const_iterator begin()const noexcept;
        constexpr inline const_iterator cbegin()const noexcept;
        constexpr inline iterator end()noexcept;
        constexpr inline const_iterator end()const noexcept;
        constexpr inline const_iterator cend()const noexcept;
        constexpr inline reverse_iterator rbegin()noexcept;
        constexpr inline const_reverse_iterator rbegin()const noexcept;
        constexpr inline const_reverse_iterator crbegin()const noexcept;
        constexpr inline reverse_iterator rend()noexcept;
        constexpr inline const_reverse_iterator rend()const noexcept;
        constexpr inline const_reverse_iterator crend()const noexcept;

        constexpr inline bool empty()const noexcept; //check if empty
        constexpr inline size_type size()const noexcept;
        // function to call instead of size() on the device

        constexpr inline size_type max_size()const noexcept;
        inline void reserve(size_type new_cap)noexcept;
        inline void grow_reserve_no_map(size_type new_cap)noexcept;
    
    // some functions to somewhat combat code duplication used to implement the other functions not meant to be called outside
    private:
        inline void buffer_resize(const size_type & new_cap)noexcept; 
        inline void buffer_resize_no_map(const size_type & new_cap)noexcept; 
    
        template<typename... Args>
        inline iterator insert_one(const_iterator pos,Args && ...args)noexcept;

        template<typename... Args>
        inline void append(Args && ...args)noexcept;        //the code for push_back and emplace_back doesn't map to device memory

        // resize except don't copy memory to device
        template<typename... Args>
        inline void resize_arr(size_type & new_size, Args && ...args)noexcept;

        template<typename index_container>
        inline void setup_buffered_insert(index_container & insert_indices, size_type count);
        template<typename indices>
        inline void setup_buffered_insert(indices insert_index[], size_type count);
    public:

        constexpr inline size_type capacity()const noexcept;
        // function to call instead of size() on the device
        inline void shrink_to_fit()noexcept;

        inline void clear()noexcept;

        inline iterator insert(const_iterator pos, const value_type& value)noexcept;
        inline iterator insert(const_iterator pos, T&& value)noexcept;
        inline iterator insert(const_iterator pos, size_type count, const T& value)noexcept;
        template<typename InputIt>
        inline iterator insert(const_iterator pos, InputIt first, InputIt last)noexcept;
        inline iterator insert(const_iterator pos, std::initializer_list<T> ilist)noexcept;
        template<typename Container>
        inline auto insert(const_iterator pos,Container && container)noexcept 
            -> type_<iterator,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size())>;
       
        // use this for multiple inserts if you don't need changes in the meantime as it only moves elements once
        template<typename Container, typename index_container>      //  the insert indices will be changed afterwards to the ones post insertion
        inline auto buffered_insert(const Container & insert_elements,index_container & insert_indices)
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size()),
                decltype(std::declval<index_container>().begin()),
                decltype(std::declval<index_container>().end()),
                decltype(std::declval<index_container>().size()),
                decltype(static_cast<int>(*(std::declval<index_container>().begin())))>;
        
        template<typename Container, typename indices>      //  the insert indices will be changed afterwards to the ones post insertion
        inline auto buffered_insert(const Container & insert_elements,indices insert_indices[], size_type count)
            -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(static_cast<int>(std::declval<indices>()))>;
    
        template<typename Container, typename index_container>      //  the insert indices will be changed afterwards to the ones post insertion
        inline auto buffered_insert(Container && insert_elements,index_container & insert_indices)
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size()),
                decltype(std::declval<index_container>().begin()),
                decltype(std::declval<index_container>().end()),
                decltype(std::declval<index_container>().size()),
                decltype(static_cast<int>(*(std::declval<index_container>().begin())))>;
        
        template<typename Container, typename indices>      //  the insert indices will be changed afterwards to the ones post insertion
        inline auto buffered_insert(Container && insert_elements,indices insert_indices[], size_type count)
            -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(static_cast<int>(std::declval<indices>()))>;


        template<typename index_container>
        inline auto buffered_erase(index_container & erase_indices)
            -> type_<void,
                decltype(std::declval<index_container>().begin()),
                decltype(std::declval<index_container>().end()),
                decltype(std::declval<index_container>().size()),
                decltype(static_cast<int>(*(std::declval<index_container>().begin())))>;
        
        template<typename indices>
        inline auto buffered_erase(indices erase_indices[], size_type count)
            -> type_<void,
                decltype(static_cast<int>(std::declval<indices>()))>;

        inline iterator erase(const_iterator pos)noexcept;
        inline iterator erase(const_iterator first, const_iterator last)noexcept;

        inline void push_back(const_reference value)noexcept;
        inline void push_back(T&& value)noexcept;

        template<typename ... Args >
        inline iterator emplace(const_iterator pos, Args&&... args)noexcept;       //both emplace and emplace back do not map changes to device memory

        template<typename... Args>
        inline reference emplace_back(Args && ...args)noexcept; 

        inline void pop_back()noexcept;

        inline void resize(size_type new_size)noexcept;

        inline void resize(size_type new_size,const_reference value)noexcept;
        

        // wrapper functions for omp_target_memcpy
        inline void memcpy_to_omp_dev(const size_type num_bytes, const size_type offset_bytes = 0)noexcept;        
        inline void memcpy_from_omp_dev(const size_type num_bytes, const size_type offset_bytes = 0)noexcept;       
        // these functions will only map the contents of the data buffer intervals are [begin,end), these are wrapper functions
        inline void map_data_to_omp_dev()noexcept;             
        inline void map_data_from_omp_dev()noexcept;
        inline void map_data_to_omp_dev(const size_type begin, const size_type end)noexcept;             
        inline void map_data_from_omp_dev(const size_type begin, const size_type end)noexcept;
        inline void map_data_to_omp_dev(const size_type begin)noexcept;             
        inline void map_data_from_omp_dev(const size_type begin)noexcept;

    private:
        // please don't use this elsewhere it is badly written
        template<typename Not_empty, typename Maybe_Empty>        
        struct packed_pair : public Maybe_Empty{
            Not_empty x_;
            constexpr packed_pair(): x_(Not_empty()), Maybe_Empty(){}
            
            constexpr packed_pair(const Not_empty &x,
                const Maybe_Empty &y=Maybe_Empty())noexcept: x_(x), Maybe_Empty(y){}
            
            constexpr Not_empty &x()noexcept{return x_;}
            constexpr Maybe_Empty &y()noexcept{return *this;}
            constexpr const Not_empty &x()const noexcept{return x_;}
            constexpr const Maybe_Empty &y()const noexcept{return *this;}
            constexpr friend void swap(packed_pair & a, packed_pair & b){
                using std::swap;
                swap(static_cast<Maybe_Empty &>(a),static_cast<Maybe_Empty &>(b));
                swap(a.x_,b.x_);
            }
        };
    // member variables
    protected:
        T* data_buffer_;                    // the data of the dynarray in host memory
        size_type size_;
        // the allocator will be packed in with capcaity
        packed_pair<size_type,allocator_type> cap_alloc_;  
        T* device_data_buffer_; // the copy on the default offloading device
    };

    template<typename T,typename Allocator,int dev_no>                                                    
    auto inline operator <<(std::ostream& stream, const dynarray<T,Allocator,dev_no> & dynamic_array) 
        -> type_<std::ostream&,
        decltype(std::cout<<std::declval<T>())>          
    {
        constexpr bool is_array_of_obj = std::is_class<T>::value;
        constexpr auto default_sep = (is_array_of_obj) ? ",\n":", ";
        stream << "[";
                                // size of the dynarray is a signed type
        for(intptr_t i = 0; i < dynamic_array.size()-1; ++i){
            stream << dynamic_array[i] << default_sep;
        }
        if (dynamic_array.size()){
            stream << dynamic_array[dynamic_array.size()-1];
        }
        stream << "]";
        return stream;
    }

    template<typename T,typename Allocator,int dev_no>
    void swap(dynarray<T,Allocator,dev_no>& array1, dynarray<T,Allocator,dev_no>& array2)noexcept{
        array1.swap(array2);
    }
    
    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::reference dynarray<T,Allocator,dev_no>::operator []
        (const size_type i)const noexcept{return data_buffer_[i];}

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::reference dynarray<T,Allocator,dev_no>::operator []
        (const size_type i)noexcept{return data_buffer_[i];}

    #pragma omp declare target device_type(nohost)
    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::reference dynarray<T,Allocator,dev_no>::operator ()
        (const size_type i)const noexcept{return device_data_buffer_[i];}
    #pragma omp end declare target

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::reference dynarray<T,Allocator,dev_no>::operator ()
        (const size_type i)noexcept{return device_data_buffer_[i];}

    template<typename T,typename Allocator,int dev_no>
    void dynarray<T,Allocator,dev_no>::swap(dynarray & rhs)noexcept{
        using std::swap;
        swap(this->data_buffer_,rhs.data_buffer_);
        swap(this->size_,rhs.size_);
        swap(this->cap_alloc_,rhs.cap_alloc_);
        swap(this->device_data_buffer_,rhs.device_data_buffer_);
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr dynarray<T,Allocator,dev_no>::dynarray()noexcept(noexcept(Allocator()))
        :data_buffer_(nullptr),
        size_(0),
        cap_alloc_(0),
        device_data_buffer_(nullptr)
    {
            create_dev_buffer();
            map_data_to_omp_dev();
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr dynarray<T,Allocator,dev_no>::dynarray(const Allocator& alloc)noexcept
        :data_buffer_(nullptr),
        size_(0),
        cap_alloc_(0,alloc),
        device_data_buffer_(nullptr)
    {
            create_dev_buffer();
            map_data_to_omp_dev();
    }

    template<typename T,typename Allocator,int dev_no>
    dynarray<T,Allocator,dev_no>::dynarray(size_type count, const T& value, const Allocator& alloc)noexcept
        :data_buffer_(nullptr),
        size_(count),
        cap_alloc_(count,alloc),
        device_data_buffer_(nullptr)
    {
        create_dynarr(std::forward<const T&>(value));        
    }

    template<typename T,typename Allocator,int dev_no>
    dynarray<T,Allocator,dev_no>::dynarray(size_type count,const Allocator& alloc)noexcept
        :data_buffer_(nullptr),
        size_(count),
        cap_alloc_(count,alloc),
        device_data_buffer_(nullptr)
    {
        create_dynarr();
    }

    template<typename T,typename Allocator,int dev_no>
    dynarray<T,Allocator,dev_no>::dynarray( const dynarray& other )noexcept(noexcept(Allocator()))
        :data_buffer_(nullptr),
        size_(other.size()),
        cap_alloc_(other.capacity(),
        std::allocator_traits<allocator_type>::select_on_container_copy_construction(
        other.get_allocator()))
    {
        create_dynarr(std::forward<const dynarray&>(other));
    }

    template<typename T,typename Allocator,int dev_no>
    dynarray<T,Allocator,dev_no>::dynarray( const dynarray& other, const Allocator& alloc )noexcept
        :data_buffer_(nullptr),
        size_(other.size()),
        cap_alloc_(other.capacity(),alloc),
        device_data_buffer_(nullptr)
    {
        create_dynarr(std::forward<const dynarray&>(other));
    }

    template<typename T,typename Allocator,int dev_no>
    dynarray<T,Allocator,dev_no>::dynarray( dynarray&& other)noexcept
        :data_buffer_(nullptr),
        size_(0),
        cap_alloc_(0,
        std::allocator_traits<allocator_type>::select_on_container_copy_construction(
        other.get_allocator())),
        device_data_buffer_(nullptr)
    {
        using std::swap;
        swap(*this,std::forward<dynarray&>(other));
    }

    template<typename T,typename Allocator,int dev_no>
    dynarray<T,Allocator,dev_no>::dynarray(dynarray&& other, const Allocator& alloc)noexcept
        :data_buffer_(nullptr),
        size_(other.size()),
        cap_alloc_(other.capacity(),alloc),
        device_data_buffer_(nullptr)
    {
        create_dynarr(std::forward<dynarray&&>(other));
    }

    template<typename T,typename Allocator,int dev_no>
    dynarray<T,Allocator,dev_no>::dynarray( std::initializer_list<T> init, const Allocator& alloc)noexcept
        :data_buffer_(nullptr),
        size_(init.size()),
        cap_alloc_(init.size(),alloc),
        device_data_buffer_(nullptr)
    {
        create_dynarr(std::forward<const std::initializer_list<T>&>(init));
    }

    template<typename T,typename Allocator,int dev_no>
    template< typename InputIt >
    dynarray<T,Allocator,dev_no>::dynarray( 
        type_<InputIt,
            decltype(std::declval<InputIt>().operator->(),
            std::declval<InputIt>().operator*())> first,
        InputIt last,const Allocator& alloc)noexcept
        
        :data_buffer_(nullptr),
        size_(last-first),
        cap_alloc_(last-first,alloc),
        device_data_buffer_(nullptr)
    {
        create_dynarr(first,last);
    }

    template<typename T,typename Allocator,int dev_no>
    dynarray<T,Allocator,dev_no>::~dynarray()noexcept{
        omp_target_free(device_data_buffer_, dev_no);
        if constexpr (!(bool)(std::is_fundamental_v<T>)){
            for (size_t i=0; i < size_;++i){
                std::allocator_traits<allocator_type>::destroy(cap_alloc_.y(),&data_buffer_[i]);
            }
        }
        std::allocator_traits<allocator_type>::deallocate(cap_alloc_.y(), reinterpret_cast<pointer>(data_buffer_), capacity());
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename Container,typename ...Last_resort>
    dynarray<T,Allocator,dev_no>::dynarray(const type_<Container,
                            decltype(std::declval<Container>().begin()),
                            decltype(std::declval<Container>().end()),
                            decltype(std::declval<Container>().size())> 
                        &array, const Allocator & alloc)noexcept
        :data_buffer_(nullptr),
        size_(0),
        cap_alloc_(0,alloc),
        device_data_buffer_(nullptr)
    {
        try
        {
            size_ = array.size();
            cap_alloc_.x() = array.size();
        }
        catch(...){
            std::cerr<<"Dynarray can only be constructed with containers that have a size() member function and support ranged based for"<<std::endl;
        }

        create_dynarr(std::forward<const Container&>(array));
    }

    template<typename T,typename Allocator,int dev_no>
    template<int dev_no2>
    dynarray<T,Allocator,dev_no>& dynarray<T,Allocator,dev_no>::operator =(const dynarray<T,Allocator,dev_no2> other){
        destroy_dealloc();
        dynarray<T,Allocator,dev_no> temp(other,
        std::allocator_traits<allocator_type>::select_on_container_copy_construction(
        other.get_allocator()));
        using std::swap;
        swap(*this,temp);
        return *this;
    }

    template<typename T,typename Allocator,int dev_no>
    dynarray<T,Allocator,dev_no>& dynarray<T,Allocator,dev_no>::operator =(const dynarray & other)noexcept{
        if (capacity()>=other.size()){
            destroy_elements();
            size_=other.size();
            construct_elements(other);
        }else{
            destroy_dealloc();
            dynarray<T,Allocator,dev_no> temp(other,
            std::allocator_traits<allocator_type>::select_on_container_copy_construction(
            other.get_allocator()));
            using std::swap;
            swap(*this,temp);
        }
        return *this;
    }

    template<typename T,typename Allocator,int dev_no>
    dynarray<T,Allocator,dev_no>& dynarray<T,Allocator,dev_no>::operator =(dynarray && other)noexcept{
        destroy_dealloc();
        using std::swap;
        swap(*this,std::forward<dynarray&>(other));
        return *this;
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename... Args>
    inline void dynarray<T,Allocator,dev_no>::create_dynarr(Args && ...args)noexcept{
        auto temp = std::allocator_traits<allocator_type>::allocate(cap_alloc_.y(),capacity());
        if (temp){
            data_buffer_ = reinterpret_cast<T*>(temp);
            construct_elements(std::forward<Args>(args)...);
        }else{
            if (capacity()>0)
            {
                std::cerr<<"ERROR dynarray creation failed to allocate memory"<<std::endl;
                if (capacity()>max_size())
                {
                    std::cerr<<"Dynarray attempted to allocate memory greater than max_size"<<"\n";
                    std::cerr<<"max_size is:"<<max_size()<<" attempted to allocate capacity of:"<<capacity()<<"\n";
                }
                size_ = 0;
                cap_alloc_.x() = 0;
                data_buffer_ = nullptr;
            }
        }
        create_dev_buffer();
        map_data_to_omp_dev();
    }
    
    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::construct_elements(const T & value)noexcept{
        difference_type it=-1;
        try
        {
            for (size_type i = 0; i < size_; ++i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),
                                &data_buffer_[i],value);
            }
        }
        catch(...)
        {
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }
    
    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::construct_elements(const size_type begin, const size_type end, const_reference value)noexcept{
        difference_type it=-1;
        try
        {
            for (difference_type i = begin; i < end; ++i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),
                                &data_buffer_[i], std::forward<const_reference>(value)); it=i;
            }
        }
        catch(...)
        {
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::construct_elements()noexcept{
        difference_type it=-1;
        try
        {
            for (size_type i = 0; i < size_; ++i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),
                                &data_buffer_[i]);
            }
        }
        catch(...)
        {
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::construct_elements(const size_type begin, const size_type end)noexcept{
        difference_type it=-1;
        try
        {
            for (difference_type i = begin; i < end; ++i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),
                                &data_buffer_[i]); it=i;
            }
        }
        catch(...)
        {
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename Container>
    inline auto dynarray<T,Allocator,dev_no>::construct_elements(const Container & container)noexcept 
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())>
    {
        size_type ptr = 0;
        difference_type it=-1;
        try
        {
            for (const auto & i : container){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),&data_buffer_[ptr],i);++ptr;
            }
        }
        catch(...)
        {
            std::cerr<<"Failed to construct elements of dynarray with ranged based for on Container, does the container support rnage based for?"<<std::endl;
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename Container>
    inline auto dynarray<T,Allocator,dev_no>::construct_elements(Container && container)noexcept 
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())>
    {
        size_type ptr = 0;
        difference_type it=-1;
        try
        {
            for (auto & i : container){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),&data_buffer_[ptr],std::move(i));++ptr;
            }
        }
        catch(...)
        {
            std::cerr<<"Failed to construct elements of dynarray with ranged based for on Container, does the container support rnage based for?"<<std::endl;
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename Container>
    inline auto dynarray<T,Allocator,dev_no>::construct_elements(const size_type begin, const size_type end,
        const Container & container)noexcept 
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())>{
        difference_type it=-1;
        try
        {
            for (difference_type i = begin; i < end; ++i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),&data_buffer_[i],container[i]);
            }
        }
        catch(...)
        {
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }
    
    template<typename T,typename Allocator,int dev_no>
    template< typename InputIt >
    inline auto dynarray<T,Allocator,dev_no>::construct_elements(InputIt first, InputIt last)noexcept 
        -> type_<void,
            decltype(*std::declval<InputIt>()),
            decltype(++std::declval<InputIt>()),
            decltype(first != last)
            >{
        size_type ptr = 0;
        difference_type it=-1;
        try
        {
            for (auto iter=first;iter!=last;++iter){std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),&data_buffer_[ptr],*iter);++ptr;}
        }
        catch(...)
        {
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::destroy_elements()noexcept{
        if constexpr (!(bool)(std::is_fundamental_v<T>)){
            difference_type it=-1;
            try
            {
                for (difference_type i=size_; i >= 0;--i){
                    std::allocator_traits<allocator_type>::destroy(cap_alloc_.y(),&data_buffer_[i]);
                    it=i;
                }
            }
            catch(...)
            {
                size_=it & -static_cast<difference_type>(it>-1);
                std::exception_ptr exception=std::current_exception();
                cout_exception(exception);
            }
        }
    }
    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::destroy_elements(const size_type new_size,const size_type old_size)noexcept{
        if constexpr (!(bool)(std::is_fundamental_v<T>)){
            difference_type it=-1;
            try
            {
                for (difference_type i=old_size; i >= new_size;--i){
                    std::allocator_traits<allocator_type>::destroy(cap_alloc_.y(),&data_buffer_[i]);
                    it=i;
                }
            }
            catch(...)
            {
                size_=it & -static_cast<difference_type>(it>-1);
                std::exception_ptr exception=std::current_exception();
                cout_exception(exception);
            }
        }
        size_ = new_size;
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::destroy_dealloc()noexcept{
        omp_target_free(device_data_buffer_, dev_no);
        device_data_buffer_ = nullptr;
        destroy_elements();
        size_=0;
        std::allocator_traits<allocator_type>::deallocate(cap_alloc_.y(), reinterpret_cast<pointer>(data_buffer_), capacity());
        cap_alloc_.x() = 0;
        data_buffer_ = nullptr;
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::create_dev_buffer()noexcept{
        T * temp = (T *)  omp_target_alloc(capacity() * sizeof(*data_buffer_), dev_no);
        if (temp){
            device_data_buffer_ = temp;
        }else if(capacity()>0){
            std::cerr<<"ERROR dynarray creation failed to allocate memory on offload device,\ndo not define TARGET_OMP_DEV macro for dynarray if not offloading with openmp"
            <<",\n ensure the device has enough memory available"<<std::endl;
        }
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::dev_buffer_reinit()noexcept{
        T * temp = (T *)  omp_target_alloc(capacity() * sizeof(*data_buffer_), dev_no);
        if (temp){
            omp_target_free(device_data_buffer_,dev_no);
            device_data_buffer_ = temp;
        }else{
            std::cerr<<"ERROR dynarray creation failed to allocate memory on offload device,\n"
            <<"do not define TARGET_OMP_DEV macro for dynarray if not offloading with openmp, ensure device has enough available memory"<<std::endl;
        }
    }


    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::assign(size_type count, const_reference value)noexcept{
        if (capacity()>=count){
            destroy_elements();
            size_=count;
            construct_elements(std::forward<const_reference>(value));
        }else{
            destroy_dealloc();
            dynarray<T,Allocator,dev_no> temp(count,std::forward<const_reference>(value),get_allocator());
            using std::swap;
            swap(*this,temp);
        }
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename InputIt >
    inline void dynarray<T,Allocator,dev_no>::assign( 
            type_<InputIt,
                decltype(std::declval<InputIt>().operator->(),
                std::declval<InputIt>().operator*())> first,
            InputIt last)noexcept{
        const difference_type count = std::distance(first,last);
        if (capacity()>=count){
            destroy_elements();
            size_=count;
            construct_elements(std::forward<InputIt>(first),std::forward<InputIt>(last));
        }else{
            destroy_dealloc();
            dynarray<T,Allocator,dev_no> temp(first,last,get_allocator());
            using std::swap;
            swap(*this,temp);
        }
    } 

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::assign(std::initializer_list<T> ilist)noexcept{
        if (capacity()>=ilist.size()){
            destroy_elements();
            size_=ilist.size();
            construct_elements(std::forward<std::initializer_list<T>>(ilist));
        }else{
            destroy_dealloc();
            dynarray<T,Allocator,dev_no> temp(std::forward<std::initializer_list<T>>(ilist),get_allocator());
            using std::swap;
            swap(*this,temp);
        }
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::allocator_type dynarray<T,Allocator,dev_no>::get_allocator()const noexcept{
        return cap_alloc_.y();
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::reference dynarray<T,Allocator,dev_no>::at(size_type pos){
        if ((pos>=size_) || (pos<0)){
            std::cerr<<"Error Dynarray indexing with at() out of bounds"
            throw std::out_of_range("Bad pos passed to at()");
        }
        else{
            return data_buffer_[pos];
        }
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::const_reference dynarray<T,Allocator,dev_no>::at(size_type pos)const{
        if ((pos>=size_) || (pos<0)){
            std::cerr<<"Error Dynarray indexing with at() out of bounds"
            throw std::out_of_range("Bad pos passed to at()");
        }
        else{
            return data_buffer_[pos];
        }
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::reference dynarray<T,Allocator,dev_no>::front(){
        return data_buffer_[0];
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::const_reference dynarray<T,Allocator,dev_no>::front()const{
        return data_buffer_[0];
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::reference dynarray<T,Allocator,dev_no>::back(){
        return data_buffer_[size_];
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::const_reference dynarray<T,Allocator,dev_no>::back()const{
        return data_buffer_[size_];
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline T* dynarray<T,Allocator,dev_no>::data()noexcept{
        return data_buffer_;
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline const T* dynarray<T,Allocator,dev_no>::data()const noexcept{
        return data_buffer_;
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline T* dynarray<T,Allocator,dev_no>::data_dev()noexcept{
        return device_data_buffer_;
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::iterator dynarray<T,Allocator,dev_no>::begin()noexcept{
        return iterator(data_buffer_);
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::const_iterator dynarray<T,Allocator,dev_no>::begin()const noexcept{
        return const_iterator(data_buffer_);
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::const_iterator dynarray<T,Allocator,dev_no>::cbegin()const noexcept{
        return const_iterator(data_buffer_);
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::iterator dynarray<T,Allocator,dev_no>::end()noexcept{
        return iterator(data_buffer_+size_);
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::const_iterator dynarray<T,Allocator,dev_no>::end()const noexcept{
        return const_iterator(data_buffer_+size_);
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::const_iterator dynarray<T,Allocator,dev_no>::cend()const noexcept{
        return const_iterator(data_buffer_+size_);
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::reverse_iterator dynarray<T,Allocator,dev_no>::rbegin()noexcept{
        return reverse_iterator(iterator(data_buffer_));
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::const_reverse_iterator dynarray<T,Allocator,dev_no>::rbegin()const noexcept{
        return const_reverse_iterator(const_iterator(data_buffer_));
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::const_reverse_iterator dynarray<T,Allocator,dev_no>::crbegin()const noexcept{
        return const_reverse_iterator(const_iterator(data_buffer_));
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::reverse_iterator dynarray<T,Allocator,dev_no>::rend()noexcept{
        return reverse_iterator(iterator(data_buffer_+size_));
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::const_reverse_iterator dynarray<T,Allocator,dev_no>::rend()const noexcept{
        return const_reverse_iterator(const_iterator(data_buffer_+size_));
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::const_reverse_iterator dynarray<T,Allocator,dev_no>::crend()const noexcept{
        return const_reverse_iterator(const_iterator(data_buffer_+size_));
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline bool dynarray<T,Allocator,dev_no>::empty()const noexcept{
        return (size_ == 0);
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::size_type dynarray<T,Allocator,dev_no>::size()const noexcept{
        return size_;
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::size_type dynarray<T,Allocator,dev_no>::max_size()const noexcept{
        return (std::numeric_limits<size_type>::max()/sizeof(T))  -1;       
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::reserve(size_type new_cap)noexcept{
        if (new_cap > capacity())
        {
            buffer_resize(new_cap);
        }
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::grow_reserve_no_map(size_type new_cap)noexcept{
        if (new_cap > capacity()){
            const size_type s = (capacity() * HOPELESS_DYNARRAY_CAPACITY_GROWTH_RATE);
            const bool sufficient_growth = s>new_cap;
            new_cap = sufficient_growth * s + static_cast<size_type>(!sufficient_growth) * new_cap;         //will have to profile this versus using if else or ternary
            buffer_resize_no_map(new_cap);
        }
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::buffer_resize(const size_type & new_cap)noexcept{
        buffer_resize_no_map(new_cap);
    #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
        map_data_to_omp_dev();
    #endif
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::buffer_resize_no_map(const size_type & new_cap)noexcept{
        auto temp = reinterpret_cast<T*>(std::allocator_traits<allocator_type>::allocate(cap_alloc_.y(),new_cap));
        if (temp){
            try{
                for (size_type i = 0; i < size_;++i){
                    std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),&temp[i],std::move(data_buffer_[i]));
                }
                destroy_elements();
                std::allocator_traits<allocator_type>::deallocate(cap_alloc_.y(), reinterpret_cast<pointer>(data_buffer_), capacity());
                data_buffer_ = temp;
                cap_alloc_.x() = new_cap;
                dev_buffer_reinit();
            }
            catch(...)
            {
                std::cout << "Dynarray failed to resize" << '\n';
                std::exception_ptr exception=std::current_exception();
                cout_exception(exception);
            }
        }else{
            if (new_cap>0)
            {
                std::cerr<<"ERROR dynarray failed to allocate memory"<<std::endl;
                if (new_cap>max_size())
                {
                    std::cerr<<"Dynarray attempted to allocate memory greater than max_size"<<"\n";
                    std::cerr<<"max_size is:"<<max_size()<<" attempted to allocate capacity of:"<<capacity()<<"\n";
                }
            }
        }
    }

    template<typename T,typename Allocator,int dev_no>  
    template<typename... Args>
    inline dynarray<T,Allocator,dev_no>::iterator dynarray<T,Allocator,dev_no>::insert_one(const_iterator pos,Args && ...args)noexcept{
        const difference_type offset = pos - begin();                                   
        grow_reserve_no_map(size() + 1);    //invalidates iterators
        pos = begin() + offset;
        try{
            if (pos != end()){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+size(),std::move(data_buffer_[size()-1]));
                size_+=1;
                for (iterator i = end()-2;i != pos; --i){
                    *i = *(i-1);
                }
                data_buffer_[offset] = T(std::forward<Args>(args)...);
            }else{
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+size(),std::forward<Args>(args)...);
                size_+=1;
            }
        #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
            map_data_to_omp_dev(offset,size());
        #endif
        }
        catch(...){
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
        return rand_access_iterator(pos.ptr());
    }

    template<typename T,typename Allocator,int dev_no>  
    template<typename... Args>
    inline void dynarray<T,Allocator,dev_no>::append(Args && ...args)noexcept{
        grow_reserve_no_map(size() + 1);    //invalidates iterators
        try
        {
            std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+size(),std::forward<Args>(args)...);
            size_+=1;
        }catch(...){
            std::cerr << "Hopeless dynarray failed to construct element" << '\n';
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator,int dev_no>  
    template<typename... Args>
    inline void dynarray<T,Allocator,dev_no>::resize_arr(size_type & new_size,Args && ... args)noexcept{
        grow_reserve_no_map(new_size);
        if (new_size < size()){
            destroy_elements(new_size,size());
        }else{
            construct_elements(size(),new_size,std::forward<Args>(args)...);
            size_ = new_size;
        }
    }

    template<typename T,typename Allocator,int dev_no> 
    template<typename index_container>
    inline void dynarray<T,Allocator,dev_no>::setup_buffered_insert(index_container & insert_indices, size_type count){
        const size_type old_size = size();
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<size_type> s_allocator_type;
        s_allocator_type s_alloc(cap_alloc_.y());
        size_type * arr_indx = reinterpret_cast<size_type*>(std::allocator_traits<s_allocator_type>::allocate(s_alloc,old_size));
        resize(old_size+count);
        for (size_type i = 0; i < old_size; ++i){
            arr_indx[i] = i;
        }
        #pragma omp parallel for
        for (size_type i = 0; i < old_size; ++i){
            for (const auto j:insert_indices){
                arr_indx[i]+=static_cast<size_type>((arr_indx[i]>=j));
            }
        }
        for (auto it = insert_indices.begin(); it != insert_indices.end(); ++it){
            for (auto other_it = insert_indices.begin();other_it != it; ++other_it){
                *other_it += (*other_it>=*it);
            }
        }
        for (difference_type i = old_size-1; i >= 0; --i){
            data_buffer_[arr_indx[i]] = std::move(data_buffer_[i]);
        }
        std::allocator_traits<s_allocator_type>::deallocate(s_alloc,reinterpret_cast<typename s_allocator_type::pointer>(arr_indx),old_size);
    }

    template<typename T,typename Allocator,int dev_no> 
    template<typename indices>
    inline void dynarray<T,Allocator,dev_no>::setup_buffered_insert(indices insert_indices[], size_type count){
        const size_type old_size = size();
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<size_type> s_allocator_type;
        s_allocator_type s_alloc(cap_alloc_.y());
        size_type * arr_indx = reinterpret_cast<size_type*>(std::allocator_traits<s_allocator_type>::allocate(s_alloc,old_size));
        resize(old_size+count);
        for (size_type i = 0; i < old_size; ++i){
            arr_indx[i] = i;
        }
        #pragma omp parallel for
        for (size_type i = 0; i < old_size; ++i){
            for (size_type j = 0 ; j < count ; ++j){
                arr_indx[i]+=static_cast<size_type>((arr_indx[i]>=insert_indices[j]));
            }
        }
        for (size_type i = 0; i < count; ++i){
            #pragma omp parallel for
            for (size_type j = 0; j < i; ++j){
                insert_indices[j] += (insert_indices[j]>=insert_indices[i]);
            }
        }
        for (difference_type i = old_size-1; i >= 0; --i){
            data_buffer_[arr_indx[i]] = std::move(data_buffer_[i]);
        }
        std::allocator_traits<s_allocator_type>::deallocate(s_alloc,reinterpret_cast<typename s_allocator_type::pointer>(arr_indx),old_size);
    }

    template<typename T,typename Allocator,int dev_no>
    constexpr inline dynarray<T,Allocator,dev_no>::size_type dynarray<T,Allocator,dev_no>::capacity()const noexcept{
        return cap_alloc_.x();       
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::shrink_to_fit()noexcept{
        buffer_resize(size(),true);
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::clear()noexcept{
        destroy_elements(0,size());
    }

    template<typename T,typename Allocator,int dev_no>
    inline dynarray<T,Allocator,dev_no>::iterator dynarray<T,Allocator,dev_no>::insert(const_iterator pos, const value_type& value)noexcept{    
        return insert_one(pos,std::forward<const T&>(value));;
    }

    template<typename T,typename Allocator,int dev_no>
    inline dynarray<T,Allocator,dev_no>::iterator dynarray<T,Allocator,dev_no>::insert(const_iterator pos, T&& value)noexcept{
        return insert_one(pos,std::forward<const T&>(value));;
    }
    template<typename T,typename Allocator,int dev_no>
    inline dynarray<T,Allocator,dev_no>::iterator dynarray<T,Allocator,dev_no>::insert(const_iterator pos, size_type count, const T& value)noexcept{
        const difference_type offset = pos - begin();   // using this to keep track (bookeep in case of iterator invalidation)                                 
        grow_reserve_no_map(size() + count);        // may invalidate iterators
        pos = begin() + offset;                     // using offset we can get back a iterator that points the correct element
        try
        {
            const difference_type num_move_c =((end()-pos)<count) ? (end()-pos):count;      // how many elements at the end of the dynarray to move construct
            for (difference_type i = size_-1; i >=  size_-num_move_c; --i){                 // everything from the last element down to the number of elements move construct
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i+count,std::move(data_buffer_[i]));
            }
            for (difference_type i = size_;i < size_ + count - num_move_c; ++i)             // the remainder of constructions up to count are thus copy constructions of value
            {
               std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i,value);
            }
            for (difference_type i = size_-count -1; i >= offset; --i)          // loop for moving rest of the elements the elements from previous size
            {
                data_buffer_[i+count] = std::move(data_buffer_[i]);
            }
            for (difference_type i = 0;i < num_move_c; ++i)                                 // filling in the reset
            {   
               data_buffer_[offset+i]=value;
            }
            size_ += count;
        #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
            map_data_to_omp_dev(offset,size());
        #endif
        }
        catch(...)
        {
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
        return rand_access_iterator(pos.ptr());
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename InputIt>
    inline dynarray<T,Allocator,dev_no>::iterator dynarray<T,Allocator,dev_no>::insert(const_iterator pos, InputIt first, InputIt last)noexcept{
        const difference_type offset = pos - begin();                   // using this to keep track (bookeep in case of iterator invalidation)
        const difference_type count = std::distance(first,last);        // total number of elements to construct                                
        grow_reserve_no_map(size() + count);        // may invalidate iterators
        pos = begin() + offset;                     // using offset we can get back a iterator that points the correct element
        try
        {
            const difference_type num_move_c =((end()-pos)<count) ? (end()-pos):count;      
            for (difference_type i = size_-1; i >=  size_-num_move_c; --i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i+count,std::move(data_buffer_[i]));
            }
            for (difference_type i = size_ + count - num_move_c-1;i >= size_; --i){
                --last;                     // intervale is [begin,end) so decrement first before dereferencing (don't want to dereference a past the end iterator)
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i,*last);
            }
            for (difference_type i =  size_-count-1; i >= offset; --i){       //loop for moving the other displaced elements
                data_buffer_[i+count] = std::move(data_buffer_[i]);
            }
            for (difference_type i = 0;i < num_move_c; ++i)
            {
                data_buffer_[offset+i] = *first;
                ++first;
            }
            size_ += count;
        #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
            map_data_to_omp_dev(offset,size());
        #endif
        }
        catch(...)
        {
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
        return rand_access_iterator(pos.ptr());
    }

    template<typename T,typename Allocator,int dev_no>
    inline dynarray<T,Allocator,dev_no>::iterator dynarray<T,Allocator,dev_no>::insert(const_iterator pos, std::initializer_list<T> ilist)noexcept{
        const difference_type offset = pos - begin();                   // using this to keep track (bookeep in case of iterator invalidation)
        const difference_type count = ilist.size();        // total number of elements to construct
        grow_reserve_no_map(size() + count);        // may invalidate iterators
        pos = begin() + offset;                     // using offset we can get back a iterator that points the correct element
        try
        {
            const difference_type num_move_c =((end()-pos)<count) ? (end()-pos):count;      
            for (difference_type i = size_-1; i >=  size_-num_move_c; --i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i+count,std::move(data_buffer_[i]));
            }
            auto ilist_last = ilist.end();
            for (difference_type i = size_ + count - num_move_c-1;i >= size_; --i){
                --ilist_last;                
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i,*(ilist_last));
            }
            for (difference_type i =  size_-count-1; i >= offset; --i){       //loop for moving the other displaced elements
                data_buffer_[i+count] = std::move(data_buffer_[i]);
            }
            for (difference_type i = 0;i < num_move_c; ++i)
            {
                data_buffer_[offset+i] = *(ilist.begin()+i);
            }
            size_ += count;
        #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
            map_data_to_omp_dev(offset,size());
        #endif
        }
        catch(...)
        {
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
        return rand_access_iterator(pos.ptr());
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename Container>
    inline auto dynarray<T,Allocator,dev_no>::insert(const_iterator pos,Container && container)noexcept
    -> type_<iterator,
        decltype(std::declval<Container>().begin()),
        decltype(std::declval<Container>().end()),
        decltype(std::declval<Container>().size())>
    {
        const difference_type offset = pos - begin();                   // using this to keep track (bookeep in case of iterator invalidation)
        const difference_type count = container.size();        // total number of elements to construct
        if ((size()+count)>capacity())                               
        {                                   
            grow_reserve_no_map(size() + count);        // invalidates iterators
            pos = begin() + offset;                     // using offset we can get back a iterator that points the correct element
        }
        try
        {
            const difference_type num_move_c =((end()-pos)<count) ? (end()-pos):count;   
            for (difference_type i = size_-1; i >=  size_-num_move_c; --i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i+count,std::move(data_buffer_[i]));
            } 
            auto container_last = container.end()-1;
            for (difference_type i = size_ + count - num_move_c-1;i >= size_; --i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i,std::move(*(container_last)));
                --container_last;
            }       
            for (difference_type i =  size_-count-1; i >= offset; --i){       //loop for moving the other displaced elements
                data_buffer_[i+count] = std::move(data_buffer_[i]);
            }
            for (difference_type i = 0;i < num_move_c; ++i)
            {
                data_buffer_[offset+i] = *(container.begin()+i);
            }
            size_ += count;
        #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
            map_data_to_omp_dev(offset,size());
        #endif
        }
        catch(...)
        {
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
        return rand_access_iterator(pos.ptr());
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename Container, typename index_container>
    inline auto dynarray<T,Allocator,dev_no>::buffered_insert(const Container & insert_elements,index_container & insert_indices)
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(std::declval<index_container>().begin()),
            decltype(std::declval<index_container>().end()),
            decltype(std::declval<index_container>().size()),
            decltype(static_cast<int>(*(std::declval<index_container>().begin())))>
    {
        size_type count = std::distance(insert_elements.begin(),insert_elements.end());
        count = (count <=  std::distance(insert_indices.begin(),insert_indices.end())) ? count:std::distance(insert_indices.begin(),insert_indices.end());
        setup_buffered_insert(insert_indices,count);
        auto elit = insert_elements.begin();
        auto it = insert_indices.begin();
        for (size_type i = 0; i < count; ++i){
            data_buffer_[*it] = *elit;
            ++elit;
            ++it;
        }
        #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
            map_data_to_omp_dev(offset,size());
        #endif
    }
    
    template<typename T,typename Allocator,int dev_no>
    template<typename Container, typename indices>
    inline auto dynarray<T,Allocator,dev_no>::buffered_insert(const Container & insert_elements,indices insert_indices[], size_type count)
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(static_cast<int>(std::declval<indices>()))>
    {
        setup_buffered_insert(insert_indices,count);
        auto elit = insert_elements.begin();
        for (size_type i = 0; i < count; ++i){
            data_buffer_[insert_indices[i]] = *elit;
            ++elit;
        }
        #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
            map_data_to_omp_dev(offset,size());
        #endif
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename Container, typename index_container>
    inline auto dynarray<T,Allocator,dev_no>::buffered_insert(Container && insert_elements,index_container & insert_indices)
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(std::declval<index_container>().begin()),
            decltype(std::declval<index_container>().end()),
            decltype(std::declval<index_container>().size()),
            decltype(static_cast<int>(*(std::declval<index_container>().begin())))>
    {
        size_type count = std::distance(insert_elements.begin(),insert_elements.end());
        count = (count <=  std::distance(insert_indices.begin(),insert_indices.end())) ? count:std::distance(insert_indices.begin(),insert_indices.end());
        setup_buffered_insert(insert_indices,count);
        auto elit = insert_elements.begin();
        auto it = insert_indices.begin();
        for (size_type i = 0; i < count; ++i){
            data_buffer_[*it] = std::move(*elit);
            ++elit;
            ++it;
        }
        #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
            map_data_to_omp_dev(offset,size());
        #endif
    }
    
    template<typename T,typename Allocator,int dev_no>
    template<typename Container, typename indices>
    inline auto dynarray<T,Allocator,dev_no>::buffered_insert(Container && insert_elements,indices insert_indices[], size_type count)
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(static_cast<int>(std::declval<indices>()))>
    {
        setup_buffered_insert(insert_indices,count);
        auto elit = insert_elements.begin();
        for (size_type i = 0; i < count; ++i){
            data_buffer_[insert_indices[i]] = std::move(*elit);
            ++elit;
        }
        #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
            map_data_to_omp_dev(offset,size());
        #endif
    }


    template<typename T,typename Allocator,int dev_no>
    template<typename index_container>
    inline auto dynarray<T,Allocator,dev_no>::buffered_erase(index_container & erase_indices)
        -> type_<void,
            decltype(std::declval<index_container>().begin()),
            decltype(std::declval<index_container>().end()),
            decltype(std::declval<index_container>().size()),
            decltype(static_cast<int>(*(std::declval<index_container>().begin())))>
    {
        size_type count = std::distance(erase_indices.begin(),erase_indices.end());
        const size_type old_size = size();
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<difference_type> d_allocator_type;
        d_allocator_type d_alloc(cap_alloc_.y());
        difference_type * arr_indx = reinterpret_cast<difference_type*>(std::allocator_traits<d_allocator_type>::allocate(d_alloc,old_size));
        for (size_type i = 0; i < size(); ++i){
            arr_indx[i] = i;
        }
        #pragma omp parallel for
        for (size_type j = 0; j < size(); ++j){
            for (const auto &i:erase_indices){
                arr_indx[j] = (arr_indx[j]==i) ? -1:arr_indx[j];
                arr_indx[j]-=static_cast<size_type>((arr_indx[j]>i));
            }
        }
        for (difference_type i = 0; i < size(); ++i){
            if (arr_indx[i] != -1)
            {
                data_buffer_[arr_indx[i]] = std::move(data_buffer_[i]);
            }
        }
        const size_type new_size = size() - count;
        resize(new_size);
        std::allocator_traits<d_allocator_type>::deallocate(d_alloc,reinterpret_cast<typename d_allocator_type::pointer>(arr_indx),old_size);
        #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
            map_data_to_omp_dev(offset,size());
        #endif
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename indices>
    inline auto dynarray<T,Allocator,dev_no>::buffered_erase(indices erase_indices[], size_type count)
        -> type_<void,
            decltype(static_cast<int>(std::declval<indices>()))>
    {
        const size_type old_size = size();
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<difference_type> d_allocator_type;
        d_allocator_type d_alloc(cap_alloc_.y());
        difference_type * arr_indx = reinterpret_cast<difference_type*>(std::allocator_traits<d_allocator_type>::allocate(d_alloc,old_size));
        for (size_type i = 0; i < size(); ++i){
            arr_indx[i] = i;
        }
        #pragma omp parallel for
        for (size_type j = 0; j < size(); ++j){
            for (size_type i = 0 ; i < count; ++i){
                arr_indx[j] = (arr_indx[j]==erase_indices[i]) ? -1:arr_indx[j];
                arr_indx[j]-=static_cast<size_type>((arr_indx[j]>erase_indices[i]));
            }
        }
        for (difference_type i = 0; i < size(); ++i){
            if (arr_indx[i] != -1)
            {
                data_buffer_[arr_indx[i]] = std::move(data_buffer_[i]);
            }
        }
        const size_type new_size = size() - count;
        resize(new_size);
        std::allocator_traits<d_allocator_type>::deallocate(d_alloc,reinterpret_cast<typename d_allocator_type::pointer>(arr_indx),old_size);
        #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
            map_data_to_omp_dev(offset,size());
        #endif
    }

    template<typename T,typename Allocator,int dev_no>
    inline dynarray<T,Allocator,dev_no>::iterator dynarray<T,Allocator,dev_no>::erase(const_iterator pos)noexcept{
        const difference_type offset = pos - begin();
        for (difference_type i = offset; i < size_-1; ++i){
            data_buffer_[i] = std::move(data_buffer_[i+1]);
        }
        std::allocator_traits<allocator_type>::destroy(cap_alloc_.y(),data_buffer_ + size_-1);
        size_ -=1;
    #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
        map_data_to_omp_dev(offset,size());
    #endif
        return iterator(data_buffer_+offset);
    }

    template<typename T,typename Allocator,int dev_no>
    inline dynarray<T,Allocator,dev_no>::iterator dynarray<T,Allocator,dev_no>::erase(const_iterator first, const_iterator last)noexcept{
        const difference_type offset = first - begin();
        const difference_type count = last - first;
        for (difference_type i = offset; i < size_ - count; ++i){
            data_buffer_[i] = std::move(data_buffer_[i+count]);
        }
        destroy_elements(size_ - count,size_);
    #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
        map_data_to_omp_dev(offset,size());
    #endif
        return iterator(data_buffer_+offset);
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::push_back(const_reference value)noexcept{
        append(std::forward<const T&>(value));
    #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
        map_data_to_omp_dev(size()-1,size());
    #endif
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::push_back(T&& value)noexcept{
        append(std::forward<T&&>(value));
    #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
        map_data_to_omp_dev(size()-1,size());
    #endif
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename... Args>
    inline dynarray<T,Allocator,dev_no>::iterator dynarray<T,Allocator,dev_no>::emplace(const_iterator pos, Args&&...args)noexcept{
        return insert_one(pos,std::forward<Args>(args)...);
    }

    template<typename T,typename Allocator,int dev_no>
    template<typename... Args>
    inline dynarray<T,Allocator,dev_no>::reference dynarray<T,Allocator,dev_no>::emplace_back(Args && ...args)noexcept{
        append(std::forward<Args>(args)...);
        return data_buffer_[size()-1];
    }

    template<typename T, typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::pop_back()noexcept{
        // will crash and burn if the dynarray is empty obviously
        try{
            std::allocator_traits<allocator_type>::destroy(cap_alloc_.y(),&data_buffer_[size()-1]);
            --size_;
        }catch(...){
            std::cerr << "Failed to call pop_back on hopeless dynarray" << '\n';
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T, typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::resize(size_type new_size)noexcept{
        const size_type old_size = size();
        resize_arr(new_size);
    #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
        map_data_to_omp_dev((old_size<new_size)?old_size:new_size); 
    #endif
    }

    template<typename T, typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::resize(size_type new_size, const_reference value)noexcept{
        const size_type old_size = size();
        resize_arr(new_size,std::forward<const_reference>(value));
    #ifdef HOPELESS_DYNARRAY_MAP_TO_DEV_POST_CHANGE
        map_data_to_omp_dev((old_size<new_size)?old_size:new_size); //choice here do we map the default intiialised elements or not?
    #endif
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::memcpy_to_omp_dev(const size_type no_bytes, const size_type offset_bytes)noexcept{
        try{
            bool fail = omp_target_memcpy(device_data_buffer_,data_buffer_,no_bytes,offset_bytes,offset_bytes,dev_no,omp_get_initial_device());
            if(no_bytes && fail){
                throw std::runtime_error("ERROR dynarray failed to copy data to device memory");
            }
        }catch(std::runtime_error& e){
           std::cerr<<e.what()<<std::endl;
        }catch(...){
            std::cerr<<"Unexpected error, possible memory corruption?"<<std::endl;
        }
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::memcpy_from_omp_dev(const size_type no_bytes, const size_type offset_bytes)noexcept{
        try{
            bool fail = omp_target_memcpy(data_buffer_,device_data_buffer_,no_bytes,offset_bytes,offset_bytes,omp_get_initial_device(),dev_no);
            if(no_bytes && fail){
                throw std::runtime_error("ERROR dynarray failed to copy data from device memory");
            }
        }catch(std::runtime_error& e){
           std::cerr<<e.what()<<std::endl;
        }catch(...){
            std::cerr<<"Unexpected error, possible memory corruption?"<<std::endl;
        }
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::map_data_to_omp_dev()noexcept{
        memcpy_to_omp_dev(size_ * sizeof(*data_buffer_));
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::map_data_from_omp_dev()noexcept{
        memcpy_from_omp_dev(size_ * sizeof(*data_buffer_));
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::map_data_to_omp_dev(const size_type begin, const size_type end)noexcept{
        memcpy_to_omp_dev((end-begin) * sizeof(*data_buffer_), begin * sizeof(*data_buffer_));
    }

    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::map_data_from_omp_dev(const size_type begin, const size_type end)noexcept{
        memcpy_from_omp_dev((end-begin) * sizeof(*data_buffer_), begin * sizeof(*data_buffer_));
    }
    
    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::map_data_to_omp_dev(const size_type begin)noexcept{
        memcpy_to_omp_dev((size_-begin) * sizeof(*data_buffer_), begin * sizeof(*data_buffer_));
    }


    template<typename T,typename Allocator,int dev_no>
    inline void dynarray<T,Allocator,dev_no>::map_data_from_omp_dev(const size_type begin)noexcept{
        memcpy_from_omp_dev((size_-begin) * sizeof(*data_buffer_), begin * sizeof(*data_buffer_));
    }

#else
    template<typename T, typename Allocator = hopeless::allocator<T>>struct  dynarray;

    template<typename T,typename Allocator>     
    auto inline operator <<(std::ostream& stream,
        const dynarray<T,Allocator> & dynamic_array) 
            -> type_<std::ostream&, 
            decltype(std::cout<<std::declval<T>())>;  

    template<typename T,typename Allocator>
    void swap(dynarray<T,Allocator>& array1, dynarray<T,Allocator>& array2)noexcept;

    template<typename T, typename Allocator>
    struct  dynarray
    {
    public:
        // declare iterators here to use for typedef
        struct rand_access_iterator;
        struct const_rand_access_iterator;

        typedef T value_type;
        typedef Allocator allocator_type;
        typedef std::ptrdiff_t difference_type;
        typedef std::ptrdiff_t size_type;       //I'm sick of the signed and unsigned nonsene I'm on a 64bit system when are you going to use an array so big the signed one would fail?
        typedef value_type& reference;
        typedef const value_type& const_reference;
        typedef typename std::allocator_traits<allocator_type>::pointer pointer;	
        typedef typename std::allocator_traits<allocator_type>::const_pointer const_pointer;	
        typedef rand_access_iterator iterator;
        typedef const_rand_access_iterator const_iterator;
        typedef typename std::reverse_iterator<iterator> reverse_iterator;
        typedef typename std::reverse_iterator<const_iterator> const_reverse_iterator;
        
        constexpr inline reference operator [](const size_type i)const noexcept;
        constexpr inline reference operator [](const size_type i)noexcept;

        constexpr inline reference operator ()(const size_type i)const noexcept;
        constexpr inline reference operator ()(const size_type i)noexcept;

        struct rand_access_iterator  
        {
        public:
            typedef std::random_access_iterator_tag iterator_category;   
            typedef T value_type;
            typedef value_type& reference;
            typedef const value_type& const_reference;
            typedef typename allocator_type::difference_type difference_type;
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
            typedef typename allocator_type::difference_type difference_type;
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
    
        void swap(dynarray<T,Allocator>& rhs)noexcept;

        constexpr dynarray()noexcept(noexcept(Allocator()));
        constexpr explicit dynarray(const Allocator& alloc)noexcept;
        dynarray(size_type count, const T& value, const Allocator& alloc = Allocator())noexcept;
        explicit dynarray(size_type count,const Allocator& alloc = Allocator())noexcept;
        
        dynarray( const dynarray& other )noexcept(noexcept(Allocator()));
        dynarray( const dynarray& other, const Allocator& alloc )noexcept;
        dynarray( dynarray&& other)noexcept;
        dynarray( dynarray&& other, const Allocator& alloc)noexcept;
        dynarray( std::initializer_list<T> init,const Allocator& alloc = Allocator())noexcept;
    	template< typename InputIt >
        dynarray(
            type_<InputIt,
                decltype(std::declval<InputIt>().operator->(),
                std::declval<InputIt>().operator*())> first,
            InputIt last,
            const Allocator& alloc = Allocator())noexcept;
        ~dynarray()noexcept;

        template<typename Container,typename ...Last_resort>
        dynarray(const type_<Container,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())> &array,
                const Allocator & alloc = Allocator())noexcept;

        dynarray& operator =(const dynarray & other)noexcept;
        dynarray& operator =(dynarray && other)noexcept;
        template<typename... Args>
        inline void create_dynarr(Args && ...args)noexcept;
        
        // the way the elements are constructed depends on what you pass to it
        inline void construct_elements(const T & value)noexcept;
        
        inline void construct_elements()noexcept;
        
        inline void construct_elements(const size_type begin, 
            const size_type end)noexcept;

        inline void construct_elements(const size_type begin, 
            const size_type end,const_reference value)noexcept;

        // initialise from a ranged based for compatible container
        template<typename Container>
        inline auto construct_elements(const Container & container)noexcept 
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size())>;
        
        template<typename Container>
        inline auto construct_elements(Container && container)noexcept 
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())>;
        
        template<typename Container>
        inline auto construct_elements(const size_type begin,
            const size_type end,
            const Container & container)noexcept 
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size())>;
        
        // initialise using iterators pointing to a container       
        template< typename InputIt >
        inline auto construct_elements(InputIt first, InputIt last)noexcept 
            -> type_<void,
                decltype(*std::declval<InputIt>()),
                decltype(++std::declval<InputIt>()),
                decltype(first != last)
                >;  // check if it acts like an iterator

        // all constructed elements up to size_
        inline void destroy_elements()noexcept;                             // this one doesn't alter the member variable size_
        inline void destroy_elements(const size_type new_size,const size_type old_size)noexcept;    // this one does
        inline void destroy_dealloc()noexcept;                              // frees up resourcess
    public:
        // will have many of the features std::vector has
        inline void assign(size_type count, const_reference value)noexcept;       
        template<typename InputIt >
        inline void assign( 
            type_<InputIt,
                decltype(std::declval<InputIt>().operator->(),
                std::declval<InputIt>().operator*())> first,
            InputIt last )noexcept;
        inline void assign( std::initializer_list<T> ilist )noexcept;
        constexpr inline allocator_type get_allocator()const noexcept;
        constexpr inline reference at( size_type pos );
        constexpr inline const_reference at( size_type pos )const;
        constexpr inline reference front();
        constexpr inline const_reference front()const;
        constexpr inline reference back();
        constexpr inline const_reference back()const;
        constexpr inline T* data()noexcept;  //get underlying buffer
        constexpr inline const T* data()const noexcept;  //get underlying buffer

        constexpr inline iterator begin() noexcept;
        constexpr inline const_iterator begin()const noexcept;
        constexpr inline const_iterator cbegin()const noexcept;
        constexpr inline iterator end()noexcept;
        constexpr inline const_iterator end()const noexcept;
        constexpr inline const_iterator cend()const noexcept;
        constexpr inline reverse_iterator rbegin()noexcept;
        constexpr inline const_reverse_iterator rbegin()const noexcept;
        constexpr inline const_reverse_iterator crbegin()const noexcept;
        constexpr inline reverse_iterator rend()noexcept;
        constexpr inline const_reverse_iterator rend()const noexcept;
        constexpr inline const_reverse_iterator crend()const noexcept;

        constexpr inline bool empty()const noexcept; //check if empty
        constexpr inline size_type size()const noexcept;

        constexpr inline size_type max_size()const noexcept;
        inline void reserve(size_type new_cap)noexcept;
        inline void grow_reserve(size_type new_cap)noexcept;

    // some functions to somewhat combat code duplication
    private:
        inline void buffer_resize(const size_type & new_cap)noexcept; 
    
        template<typename... Args>
        inline iterator insert_one(const_iterator pos,Args && ...args)noexcept;

        template<typename... Args>
        inline void append(Args && ...args)noexcept;    

        template<typename... Args>
        inline void resize_arr(size_type & new_size, Args && ...args)noexcept;   

        template<typename index_container>
        inline void setup_buffered_insert(index_container & insert_indices, size_type count);
        template<typename indices>
        inline void setup_buffered_insert(indices insert_index[], size_type count);
    public:

        constexpr inline size_type capacity()const noexcept;
        inline void shrink_to_fit()noexcept;

        inline void clear()noexcept;

        inline iterator insert(const_iterator pos, const value_type& value)noexcept;
        inline iterator insert(const_iterator pos, T&& value)noexcept;
        inline iterator insert(const_iterator pos, size_type count, const T& value)noexcept;
        template<typename InputIt>
        inline iterator insert(const_iterator pos, InputIt first, InputIt last)noexcept;
        inline iterator insert(const_iterator pos, std::initializer_list<T> ilist)noexcept;
        template<typename Container>
        inline auto insert(const_iterator pos,Container && container)noexcept 
            -> type_<iterator,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size())>;
       

        // use this for multiple inserts if you don't need changes in the meantime as it only moves elements once
        template<typename Container, typename index_container>      //  the insert indices will be changed afterwards to the ones post insertion
        inline auto buffered_insert(const Container & insert_elements,index_container & insert_indices)
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size()),
                decltype(std::declval<index_container>().begin()),
                decltype(std::declval<index_container>().end()),
                decltype(std::declval<index_container>().size()),
                decltype(static_cast<int>(*(std::declval<index_container>().begin())))>;
        
        template<typename Container, typename indices>      //  the insert indices will be changed afterwards to the ones post insertion
        inline auto buffered_insert(const Container & insert_elements,indices insert_indices[], size_type count)
            -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(static_cast<int>(std::declval<indices>()))>;
    
        template<typename Container, typename index_container>      //  the insert indices will be changed afterwards to the ones post insertion
        inline auto buffered_insert(Container && insert_elements,index_container & insert_indices)
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size()),
                decltype(std::declval<index_container>().begin()),
                decltype(std::declval<index_container>().end()),
                decltype(std::declval<index_container>().size()),
                decltype(static_cast<int>(*(std::declval<index_container>().begin())))>;
        
        template<typename Container, typename indices>      //  the insert indices will be changed afterwards to the ones post insertion
        inline auto buffered_insert(Container && insert_elements,indices insert_indices[], size_type count)
            -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(static_cast<int>(std::declval<indices>()))>;
        
        template<typename index_container>
        inline auto buffered_erase(index_container & erase_indices)
            -> type_<void,
                decltype(std::declval<index_container>().begin()),
                decltype(std::declval<index_container>().end()),
                decltype(std::declval<index_container>().size()),
                decltype(static_cast<int>(*(std::declval<index_container>().begin())))>;
        
        template<typename indices>
        inline auto buffered_erase(indices erase_indices[], size_type count)
            -> type_<void,
                decltype(static_cast<int>(std::declval<indices>()))>;
        
        inline iterator erase(const_iterator pos)noexcept;
        inline iterator erase(const_iterator first, const_iterator last)noexcept;

        inline void push_back(const_reference value)noexcept;
        inline void push_back(T&& value)noexcept;

        template<typename ... Args >
        inline iterator emplace(const_iterator pos, Args&&... args)noexcept;       

        template<typename... Args>
        inline reference emplace_back(Args && ...args)noexcept; 

        inline void pop_back()noexcept;

        inline void resize(size_type new_size)noexcept;

        inline void resize(size_type new_size,const_reference value)noexcept;  

    private:
        template<typename Not_empty, typename Maybe_Empty>        
        struct packed_pair : public Maybe_Empty{
            Not_empty x_;
            constexpr packed_pair()noexcept{}
            
            constexpr packed_pair(const Not_empty &x=Not_empty(),
                const Maybe_Empty &y=Maybe_Empty())noexcept: x_(x), Maybe_Empty(y) {}
            
            constexpr Not_empty &x()noexcept{return x_;}
            constexpr Maybe_Empty &y()noexcept{return *this;}
            constexpr const Not_empty &x()const noexcept{return x_;}
            constexpr const Maybe_Empty &y()const noexcept{return *this;}
            constexpr friend void swap(packed_pair & a, packed_pair & b){
                using std::swap;
                swap(static_cast<Maybe_Empty &>(a),static_cast<Maybe_Empty &>(b));
                swap(a.x_,b.x_);
            }
        };
    // member variables
    protected:
        T* data_buffer_;                    // the data of the dynarray in host memory
        size_type size_;
        // the allocator will be packed in with capcaity
        packed_pair<size_type,allocator_type> cap_alloc_;
    };

    template<typename T,typename Allocator>                                                    
    auto inline operator <<(std::ostream& stream, const dynarray<T,Allocator> & dynamic_array) 
        -> type_<std::ostream&,
        decltype(std::cout<<std::declval<T>())>          
    {
        constexpr bool is_array_of_obj = std::is_class<T>::value;
        constexpr auto default_sep = (is_array_of_obj) ? ",\n":", ";
        stream << "[";
        const intptr_t up_to_last = dynamic_array.size()-1;
        for(intptr_t i = 0; i < up_to_last; ++i){
            stream << dynamic_array[i] << default_sep;
        }
        if (dynamic_array.size()){
            stream << dynamic_array[dynamic_array.size()-1];
        }
        stream << "]";
        return stream;
    }

    template<typename T,typename Allocator>
    void swap(dynarray<T,Allocator>& array1, dynarray<T,Allocator>& array2)noexcept{
        array1.swap(array2);
    }
    
    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::reference dynarray<T,Allocator>::operator []
        (const size_type i)const noexcept{return data_buffer_[i];}

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::reference dynarray<T,Allocator>::operator []
        (const size_type i)noexcept{return data_buffer_[i];}

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::reference dynarray<T,Allocator>::operator ()
        (const size_type i)const noexcept{return data_buffer_[i];}
    
    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::reference dynarray<T,Allocator>::operator ()
        (const size_type i)noexcept{return data_buffer_[i];}

    template<typename T,typename Allocator>
    void dynarray<T,Allocator>::swap(dynarray & rhs)noexcept{
        using std::swap;
        swap(this->data_buffer_,rhs.data_buffer_);
        swap(this->size_,rhs.size_);
        swap(this->cap_alloc_,rhs.cap_alloc_);
    }

    template<typename T,typename Allocator>
    constexpr dynarray<T,Allocator>::dynarray()noexcept(noexcept(Allocator()))
        :data_buffer_(nullptr),
        size_(0),
        cap_alloc_(0){}

    template<typename T,typename Allocator>
    constexpr dynarray<T,Allocator>::dynarray(const Allocator& alloc)noexcept
        :data_buffer_(nullptr),
        size_(0),
        cap_alloc_(0,alloc){}

    template<typename T,typename Allocator>
    dynarray<T,Allocator>::dynarray(size_type count, const T& value, const Allocator& alloc)noexcept
        :data_buffer_(nullptr),
        size_(count),
        cap_alloc_(count,alloc)
    {
        create_dynarr(std::forward<const T&>(value));
    }

    template<typename T,typename Allocator>
    dynarray<T,Allocator>::dynarray(size_type count,const Allocator& alloc)noexcept
        :data_buffer_(nullptr),
        size_(count),
        cap_alloc_(count,alloc){}

    template<typename T,typename Allocator>
    dynarray<T,Allocator>::dynarray( const dynarray& other )noexcept(noexcept(Allocator()))
        :data_buffer_(nullptr),
        size_(other.size()),
        cap_alloc_(other.capacity(),
        std::allocator_traits<allocator_type>::select_on_container_copy_construction(
        other.get_allocator()))
    {
        create_dynarr(std::forward<const dynarray&>(other));
    }

    template<typename T,typename Allocator>
    dynarray<T,Allocator>::dynarray( const dynarray& other, const Allocator& alloc )noexcept
        :data_buffer_(nullptr),
        size_(other.size()),
        cap_alloc_(other.capacity(),alloc)
    {
        create_dynarr(std::forward<const dynarray&>(other));
    }

    template<typename T,typename Allocator>
    dynarray<T,Allocator>::dynarray( dynarray&& other)noexcept
        :data_buffer_(nullptr),
        size_(0),
        cap_alloc_(0,
        std::allocator_traits<allocator_type>::select_on_container_copy_construction(
        other.get_allocator()))
    {
        using std::swap;
        swap(*this,std::forward<dynarray&>(other));
    }

    template<typename T,typename Allocator>
    dynarray<T,Allocator>::dynarray(dynarray&& other, const Allocator& alloc)noexcept
        :data_buffer_(nullptr),
        size_(other.size()),
        cap_alloc_(other.capacity(),alloc)
    {
        create_dynarr(std::forward<dynarray&&>(other));
    }

    template<typename T,typename Allocator>
    dynarray<T,Allocator>::dynarray( std::initializer_list<T> init, const Allocator& alloc)noexcept
        :data_buffer_(nullptr),
        size_(init.size()),
        cap_alloc_(init.size(),alloc)
    {
        create_dynarr(std::forward<const std::initializer_list<T>&>(init));
    }

    template<typename T,typename Allocator>
    template< typename InputIt >
    dynarray<T,Allocator>::dynarray( 
        type_<InputIt,
            decltype(std::declval<InputIt>().operator->(),
            std::declval<InputIt>().operator*())> first,
        InputIt last,const Allocator& alloc)noexcept
        
        :data_buffer_(nullptr),
        size_(last-first),
        cap_alloc_(last-first,alloc)
    {
        create_dynarr(first,last);
    }

    template<typename T,typename Allocator>
    dynarray<T,Allocator>::~dynarray()noexcept{
        if constexpr (!(bool)(std::is_fundamental_v<T>)){
            for (size_t i=0; i < size_;++i){
                std::allocator_traits<allocator_type>::destroy(cap_alloc_.y(),&data_buffer_[i]);
            }
        }
        std::allocator_traits<allocator_type>::deallocate(cap_alloc_.y(), reinterpret_cast<pointer>(data_buffer_), capacity());
    }

    template<typename T,typename Allocator>
    template<typename Container,typename ...Last_resort>
    dynarray<T,Allocator>::dynarray(const type_<Container,
                            decltype(std::declval<Container>().begin()),
                            decltype(std::declval<Container>().end()),
                            decltype(std::declval<Container>().size())> 
                        &array, const Allocator & alloc)noexcept
        :data_buffer_(nullptr),
        size_(0),
        cap_alloc_(0,alloc)
    {
        try
        {
            size_ = array.size();
            cap_alloc_.x() = array.size();
        }
        catch(...){
            std::cerr<<"Dynarray can only be constructed with containers that have a size() member function and support ranged based for"<<std::endl;
        }

        create_dynarr(std::forward<const Container&>(array));
    }

    template<typename T,typename Allocator>
    dynarray<T,Allocator>& dynarray<T,Allocator>::operator =(const dynarray & other)noexcept{
        if (capacity()>=other.size()){
            destroy_elements();
            size_=other.size();
            construct_elements(other);
        }else{
            destroy_dealloc();
            dynarray<T,Allocator> temp(other,
            std::allocator_traits<allocator_type>::select_on_container_copy_construction(
            other.get_allocator()));
            using std::swap;
            swap(*this,temp);
        }
        return *this;
    }

    template<typename T,typename Allocator>
    dynarray<T,Allocator>& dynarray<T,Allocator>::operator =(dynarray && other)noexcept{
        destroy_dealloc();
        using std::swap;
        swap(*this,std::forward<dynarray&>(other));
        return *this;
    }

    template<typename T,typename Allocator>
    template<typename... Args>
    inline void dynarray<T,Allocator>::create_dynarr(Args && ...args)noexcept{
        auto temp = std::allocator_traits<allocator_type>::allocate(cap_alloc_.y(),capacity());
        if (temp){
            data_buffer_ = reinterpret_cast<T*>(temp);
            construct_elements(std::forward<Args>(args)...);
        }else{
            if (capacity()>0)
            {
                std::cerr<<"ERROR dynarray creation failed to allocate memory"<<std::endl;
                if (capacity()>max_size())
                {
                    std::cerr<<"Dynarray attempted to allocate memory greater than max_size"<<"\n";
                    std::cerr<<"max_size is:"<<max_size()<<" attempted to allocate capacity of:"<<capacity()<<"\n";
                }
                size_ = 0;
                cap_alloc_.x() = 0;
                data_buffer_ = nullptr;
            }
        }
    }
    
    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::construct_elements(const T & value)noexcept{
        difference_type it=-1;
        try
        {
            for (size_type i = 0; i < size_; ++i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),
                                &data_buffer_[i],value);
            }
        }
        catch(...)
        {
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }
    
    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::construct_elements(const size_type begin, const size_type end, const_reference value)noexcept{
        difference_type it=-1;
        try
        {
            for (difference_type i = begin; i < end; ++i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),
                                &data_buffer_[i], std::forward<const_reference>(value)); it=i;
            }
        }
        catch(...)
        {
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::construct_elements()noexcept{
        difference_type it=-1;
        try
        {
            for (size_type i = 0; i < size_; ++i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),
                                &data_buffer_[i]);
            }
        }
        catch(...)
        {
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::construct_elements(const size_type begin, const size_type end)noexcept{
        difference_type it=-1;
        try
        {
            for (difference_type i = begin; i < end; ++i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),
                                &data_buffer_[i]); it=i;
            }
        }
        catch(...)
        {
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator>
    template<typename Container>
    inline auto dynarray<T,Allocator>::construct_elements(const Container & container)noexcept 
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())>{
        size_type ptr = 0;
        difference_type it=-1;
        try
        {
            for (const auto & i : container){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),&data_buffer_[ptr],i);++ptr;
            }
        }
        catch(...)
        {
            std::cerr<<"Failed to construct elements of dynarray with ranged based for on Container, does the container support rnage based for?"<<std::endl;
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator>
    template<typename Container>
    inline auto dynarray<T,Allocator>::construct_elements(Container && container)noexcept 
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())>
    {
        size_type ptr = 0;
        difference_type it=-1;
        try
        {
            for (auto & i : container){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),&data_buffer_[ptr],std::move(i));++ptr;
            }
        }
        catch(...)
        {
            std::cerr<<"Failed to construct elements of dynarray with ranged based for on Container, does the container support rnage based for?"<<std::endl;
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator>
    template<typename Container>
    inline auto dynarray<T,Allocator>::construct_elements(const size_type begin, const size_type end,
        const Container & container)noexcept 
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())>{
        difference_type it=-1;
        try
        {
            #pragma omp loop
            for (difference_type i = begin; i < end; ++i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),&data_buffer_[i],container[i]);
            }
        }
        catch(...)
        {
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }
    
    template<typename T,typename Allocator>
    template< typename InputIt >
    inline auto dynarray<T,Allocator>::construct_elements(InputIt first, InputIt last)noexcept 
        -> type_<void,
            decltype(*std::declval<InputIt>()),
            decltype(++std::declval<InputIt>()),
            decltype(first != last)
            >{
        size_type ptr = 0;
        difference_type it=-1;
        try
        {
            for (auto iter=first;iter!=last;++iter){std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),&data_buffer_[ptr],*iter);++ptr;}
        }
        catch(...)
        {
            size_=it & -static_cast<difference_type>(it>-1);
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::destroy_elements()noexcept{
        if constexpr (!(bool)(std::is_fundamental_v<T>)){
            difference_type it=-1;
            try
            {
                for (difference_type i=size_; i >= 0;--i){
                    std::allocator_traits<allocator_type>::destroy(cap_alloc_.y(),&data_buffer_[i]);
                    it=i;
                }
            }
            catch(...)
            {
                size_=it & -static_cast<difference_type>(it>-1);
                std::exception_ptr exception=std::current_exception();
                cout_exception(exception);
            }
        }
    }
    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::destroy_elements(const size_type new_size,const size_type old_size)noexcept{
        if constexpr (!(bool)(std::is_fundamental_v<T>)){
            difference_type it=-1;
            try
            {
                for (difference_type i=old_size; i >= new_size;--i){
                    std::allocator_traits<allocator_type>::destroy(cap_alloc_.y(),&data_buffer_[i]);
                    it=i;
                }
            }
            catch(...)
            {
                size_=it & -static_cast<difference_type>(it>-1);
                std::exception_ptr exception=std::current_exception();
                cout_exception(exception);
            }
        }
        size_ = new_size;
    }

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::destroy_dealloc()noexcept{
        destroy_elements();
        size_=0;
        std::allocator_traits<allocator_type>::deallocate(cap_alloc_.y(), reinterpret_cast<pointer>(data_buffer_), capacity());
        cap_alloc_.x() = 0;
        data_buffer_ = nullptr;
    }

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::assign(size_type count, const_reference value)noexcept{
        if (capacity()>=count){
            destroy_elements();
            size_=count;
            construct_elements(std::forward<const_reference>(value));
        }else{
            destroy_dealloc();
            dynarray<T,Allocator> temp(count,std::forward<const_reference>(value),get_allocator());
            using std::swap;
            swap(*this,temp);
        }
    }

    template<typename T,typename Allocator>
    template<typename InputIt >
    inline void dynarray<T,Allocator>::assign( 
            type_<InputIt,
                decltype(std::declval<InputIt>().operator->(),
                std::declval<InputIt>().operator*())> first,
            InputIt last)noexcept{
        const difference_type count = std::distance(first,last);
        if (capacity()>=count){
            destroy_elements();
            size_=count;
            construct_elements(std::forward<InputIt>(first),std::forward<InputIt>(last));
        }else{
            destroy_dealloc();
            dynarray<T,Allocator> temp(first,last,get_allocator());
            using std::swap;
            swap(*this,temp);
        }
    } 

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::assign(std::initializer_list<T> ilist)noexcept{
        if (capacity()>=ilist.size()){
            destroy_elements();
            size_=ilist.size();
            construct_elements(std::forward<std::initializer_list<T>>(ilist));
        }else{
            destroy_dealloc();
            dynarray<T,Allocator> temp(std::forward<std::initializer_list<T>>(ilist),get_allocator());
            using std::swap;
            swap(*this,temp);
        }
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::allocator_type dynarray<T,Allocator>::get_allocator()const noexcept{
        return cap_alloc_.y();
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::reference dynarray<T,Allocator>::at(size_type pos){
        if ((pos>=size_) || (pos<0)){
            std::cerr<<"Error Dynarray indexing with at() out of bounds"
            throw std::out_of_range("Bad pos passed to at()");
        }
        else{
            return data_buffer_[pos];
        }
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::const_reference dynarray<T,Allocator>::at(size_type pos)const{
        if ((pos>=size_) || (pos<0)){
            std::cerr<<"Error Dynarray indexing with at() out of bounds"
            throw std::out_of_range("Bad pos passed to at()");
        }
        else{
            return data_buffer_[pos];
        }
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::reference dynarray<T,Allocator>::front(){
        return data_buffer_[0];
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::const_reference dynarray<T,Allocator>::front()const{
        return data_buffer_[0];
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::reference dynarray<T,Allocator>::back(){
        return data_buffer_[size_];
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::const_reference dynarray<T,Allocator>::back()const{
        return data_buffer_[size_];
    }

    template<typename T,typename Allocator>
    constexpr inline T* dynarray<T,Allocator>::data()noexcept{
        return data_buffer_;
    }

    template<typename T,typename Allocator>
    constexpr inline const T* dynarray<T,Allocator>::data()const noexcept{
        return data_buffer_;
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::iterator dynarray<T,Allocator>::begin()noexcept{
        return rand_access_iterator(data_buffer_);
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::const_iterator dynarray<T,Allocator>::begin()const noexcept{
        return const_rand_access_iterator(data_buffer_);
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::const_iterator dynarray<T,Allocator>::cbegin()const noexcept{
        return const_rand_access_iterator(data_buffer_);
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::iterator dynarray<T,Allocator>::end()noexcept{
        return rand_access_iterator(data_buffer_+size_);
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::const_iterator dynarray<T,Allocator>::end()const noexcept{
        return const_rand_access_iterator(data_buffer_+size_);
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::const_iterator dynarray<T,Allocator>::cend()const noexcept{
        return const_rand_access_iterator(data_buffer_+size_);
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::reverse_iterator dynarray<T,Allocator>::rbegin()noexcept{
        return std::reverse_iterator<rand_access_iterator>(rand_access_iterator(data_buffer_));
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::const_reverse_iterator dynarray<T,Allocator>::rbegin()const noexcept{
        return std::reverse_iterator<const_rand_access_iterator>(const_rand_access_iterator(data_buffer_));
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::const_reverse_iterator dynarray<T,Allocator>::crbegin()const noexcept{
        return std::reverse_iterator<const_rand_access_iterator>(const_rand_access_iterator(data_buffer_));
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::reverse_iterator dynarray<T,Allocator>::rend()noexcept{
        return std::reverse_iterator<rand_access_iterator>(rand_access_iterator(data_buffer_+size_));
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::const_reverse_iterator dynarray<T,Allocator>::rend()const noexcept{
        return std::reverse_iterator<const_rand_access_iterator>(const_rand_access_iterator(data_buffer_+size_));
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::const_reverse_iterator dynarray<T,Allocator>::crend()const noexcept{
        return std::reverse_iterator<const_rand_access_iterator>(const_rand_access_iterator(data_buffer_+size_));
    }

    template<typename T,typename Allocator>
    constexpr inline bool dynarray<T,Allocator>::empty()const noexcept{
        return (size_ == 0);
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::size_type dynarray<T,Allocator>::size()const noexcept{
        return size_;
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::size_type dynarray<T,Allocator>::max_size()const noexcept{
        return (std::numeric_limits<size_type>::max()/sizeof(T))  -1;       
    }

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::reserve(size_type new_cap)noexcept{
        if (new_cap > capacity())
        {
            buffer_resize(new_cap);
        }
    }

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::grow_reserve(size_type new_cap)noexcept{
        if (new_cap > capacity()){
            const size_type s = (capacity() * HOPELESS_DYNARRAY_CAPACITY_GROWTH_RATE);
            const bool sufficient_growth = (capacity() * HOPELESS_DYNARRAY_CAPACITY_GROWTH_RATE)>new_cap;
            new_cap = sufficient_growth * s + static_cast<size_type>(!sufficient_growth) * new_cap;         //will have to profile this versus using if else or ternary
            buffer_resize(new_cap);
        }
    }

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::buffer_resize(const size_type & new_cap)noexcept{
        auto temp = reinterpret_cast<T*>(std::allocator_traits<allocator_type>::allocate(cap_alloc_.y(),new_cap));
        if (temp){
            try{
                for (size_type i = 0; i < size_;++i){
                    std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),&temp[i],std::move(data_buffer_[i]));
                }
                destroy_elements();
                std::allocator_traits<allocator_type>::deallocate(cap_alloc_.y(), reinterpret_cast<pointer>(data_buffer_), capacity());
                data_buffer_ = temp;
                cap_alloc_.x() = new_cap;
            }
            catch(...)
            {
                std::cout << "Dynarray failed to resize" << '\n';
                std::exception_ptr exception=std::current_exception();
                cout_exception(exception);
            }
        }else{
            if (new_cap>0)
            {
                std::cerr<<"ERROR dynarray failed to allocate memory"<<std::endl;
                if (new_cap>max_size())
                {
                    std::cerr<<"Dynarray attempted to allocate memory greater than max_size"<<"\n";
                    std::cerr<<"max_size is:"<<max_size()<<" attempted to allocate capacity of:"<<capacity()<<"\n";
                }
            }
        }
    }

    template<typename T,typename Allocator>  
    template<typename... Args>
    inline dynarray<T,Allocator>::iterator dynarray<T,Allocator>::insert_one(const_iterator pos,Args && ...args)noexcept{
        const difference_type offset = pos - begin();                                   
        grow_reserve(size() + 1);    //invalidates iterators
        pos = begin() + offset;
        try{
            if (pos != end()){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+size(),std::move(data_buffer_[size()-1]));
                size_+=1;
                for (iterator i = end()-2;i != pos; --i){
                    *i = *(i-1);
                }
                data_buffer_[offset] = T(std::forward<Args>(args)...);
            }else{
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+size(),std::forward<Args>(args)...);
                size_+=1;
            }
        }
        catch(...){
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
        return rand_access_iterator(pos.ptr());
    }

    template<typename T,typename Allocator>  
    template<typename... Args>
    inline void dynarray<T,Allocator>::append(Args && ...args)noexcept{
        grow_reserve(size() + 1);    //invalidates iterators
        try
        {
            std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+size(),std::forward<Args>(args)...);
            size_+=1;
        }catch(...){
            std::cerr << "Hopeless dynarray failed to construct element" << '\n';
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T,typename Allocator>  
    template<typename... Args>
    inline void dynarray<T,Allocator>::resize_arr(size_type & new_size,Args && ... args)noexcept{
        grow_reserve(new_size);
        if (new_size < size()){
            destroy_elements(new_size,size());
        }else{
            construct_elements(size(),new_size,std::forward<Args>(args)...);
            size_ = new_size;
        }
    }

    template<typename T,typename Allocator> 
    template<typename index_container>
    inline void dynarray<T,Allocator>::setup_buffered_insert(index_container & insert_indices, size_type count){
        const size_type old_size = size();
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<size_type> s_allocator_type;
        s_allocator_type s_alloc(cap_alloc_.y());
        size_type * arr_indx = reinterpret_cast<size_type*>(std::allocator_traits<s_allocator_type>::allocate(s_alloc,old_size));
        resize(old_size+count);
        for (size_type i = 0; i < old_size; ++i){
            arr_indx[i] = i;
        }
        #pragma omp parallel for
        for (size_type i = 0; i < old_size; ++i){
            for (const auto j:insert_indices){
                arr_indx[i]+=static_cast<size_type>((arr_indx[i]>=j));
            }
        }
        for (auto it = insert_indices.begin(); it != insert_indices.end(); ++it){
            for (auto other_it = insert_indices.begin();other_it != it; ++other_it){
                *other_it += (*other_it>=*it);
            }
        }
        for (difference_type i = old_size-1; i >= 0; --i){
            data_buffer_[arr_indx[i]] = std::move(data_buffer_[i]);
        }
        std::allocator_traits<s_allocator_type>::deallocate(s_alloc,reinterpret_cast<typename s_allocator_type::pointer>(arr_indx),old_size);
    }

    template<typename T,typename Allocator> 
    template<typename indices>
    inline void dynarray<T,Allocator>::setup_buffered_insert(indices insert_indices[], size_type count){
        const size_type old_size = size();
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<size_type> s_allocator_type;
        s_allocator_type s_alloc(cap_alloc_.y());
        size_type * arr_indx = reinterpret_cast<size_type*>(std::allocator_traits<s_allocator_type>::allocate(s_alloc,old_size));
        resize(old_size+count);
        for (size_type i = 0; i < old_size; ++i){
            arr_indx[i] = i;
        }
        #pragma omp parallel for
        for (size_type i = 0; i < old_size; ++i){
            for (size_type j = 0 ; j < count ; ++j){
                arr_indx[i]+=static_cast<size_type>((arr_indx[i]>=insert_indices[j]));
            }
        }
        for (size_type i = 0; i < count; ++i){
            #pragma omp parallel for
            for (size_type j = 0; j < i; ++j){
                insert_indices[j] += (insert_indices[j]>=insert_indices[i]);
            }
        }
        for (difference_type i = old_size-1; i >= 0; --i){
            data_buffer_[arr_indx[i]] = std::move(data_buffer_[i]);
        }
        std::allocator_traits<s_allocator_type>::deallocate(s_alloc,reinterpret_cast<typename s_allocator_type::pointer>(arr_indx),old_size);
    }

    template<typename T,typename Allocator>
    constexpr inline dynarray<T,Allocator>::size_type dynarray<T,Allocator>::capacity()const noexcept{
        return cap_alloc_.x();       
    }

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::shrink_to_fit()noexcept{
        buffer_resize(size(),true);
    }

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::clear()noexcept{
        destroy_elements(0,size());
    }

    template<typename T,typename Allocator>
    inline dynarray<T,Allocator>::iterator dynarray<T,Allocator>::insert(const_iterator pos, const value_type& value)noexcept{    
        return insert_one(pos,std::forward<const T&>(value));
    }

    template<typename T,typename Allocator>
    inline dynarray<T,Allocator>::iterator dynarray<T,Allocator>::insert(const_iterator pos, T&& value)noexcept{
        return insert_one(pos,std::forward<const T&>(value));
    }
    template<typename T,typename Allocator>
    inline dynarray<T,Allocator>::iterator dynarray<T,Allocator>::insert(const_iterator pos, size_type count, const T& value)noexcept{
        const difference_type offset = pos - begin();   // using this to keep track (bookeep in case of iterator invalidation)                                 
        grow_reserve(size() + count);        // may invalidate iterators
        pos = begin() + offset;                     // using offset we can get back a iterator that points the correct element
        try
        {
            const difference_type num_move_c =((end()-pos)<count) ? (end()-pos):count;      // how many elements at the end of the dynarray to move construct
            for (difference_type i = size_-1; i >=  size_-num_move_c; --i){                 // everything from the last element down to the number of elements move construct
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i+count,std::move(data_buffer_[i]));
            }
            for (difference_type i = size_;i < size_ + count - num_move_c; ++i)             // the remainder of constructions up to count are thus copy constructions of value
            {
               std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i,value);
            }
            for (difference_type i = size_-count -1; i >= offset; --i)          // loop for moving rest of the elements the elements from previous size
            {
                data_buffer_[i+count] = std::move(data_buffer_[i]);
            }
            for (difference_type i = 0;i < num_move_c; ++i)                                 // filling in the reset
            {   
               data_buffer_[offset+i]=value;
            }
            size_ += count;
        }
        catch(...)
        {
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
        return rand_access_iterator(pos.ptr());
    }

    template<typename T,typename Allocator>
    template<typename InputIt>
    inline dynarray<T,Allocator>::iterator dynarray<T,Allocator>::insert(const_iterator pos, InputIt first, InputIt last)noexcept{
        const difference_type offset = pos - begin();                   // using this to keep track (bookeep in case of iterator invalidation)
        const difference_type count = std::distance(first,last);        // total number of elements to construct                                
        grow_reserve(size() + count);        // may invalidate iterators
        pos = begin() + offset;                     // using offset we can get back a iterator that points the correct element
        try
        {
            const difference_type num_move_c =((end()-pos)<count) ? (end()-pos):count;      
            for (difference_type i = size_-1; i >=  size_-num_move_c; --i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i+count,std::move(data_buffer_[i]));
            }
            for (difference_type i = size_ + count - num_move_c-1;i >= size_; --i){
                --last;                     // intervale is [begin,end) so decrement first before dereferencing (don't want to dereference a past the end iterator)
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i,*last);
            }
            for (difference_type i =  size_-count-1; i >= offset; --i){       //loop for moving the other displaced elements
                data_buffer_[i+count] = std::move(data_buffer_[i]);
            }
            for (difference_type i = 0;i < num_move_c; ++i)
            {
                data_buffer_[offset+i] = *first;
                ++first;
            }
            size_ += count;
        }
        catch(...)
        {
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
        return rand_access_iterator(pos.ptr());
    }

    template<typename T,typename Allocator>
    inline dynarray<T,Allocator>::iterator dynarray<T,Allocator>::insert(const_iterator pos, std::initializer_list<T> ilist)noexcept{
        const difference_type offset = pos - begin();                   // using this to keep track (bookeep in case of iterator invalidation)
        const difference_type count = ilist.size();        // total number of elements to construct
        grow_reserve(size() + count);        // may invalidate iterators
        pos = begin() + offset;                     // using offset we can get back a iterator that points the correct element
        try
        {
            const difference_type num_move_c =((end()-pos)<count) ? (end()-pos):count;      
            for (difference_type i = size_-1; i >=  size_-num_move_c; --i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i+count,std::move(data_buffer_[i]));
            }
            auto ilist_last = ilist.end();
            for (difference_type i = size_ + count - num_move_c-1;i >= size_; --i){
                --ilist_last;                
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i,*(ilist_last));
            }
            for (difference_type i =  size_-count-1; i >= offset; --i){       //loop for moving the other displaced elements
                data_buffer_[i+count] = std::move(data_buffer_[i]);
            }
            for (difference_type i = 0;i < num_move_c; ++i)
            {
                data_buffer_[offset+i] = *(ilist.begin()+i);
            }
            size_ += count;
        }
        catch(...)
        {
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
        return rand_access_iterator(pos.ptr());
    }

    template<typename T,typename Allocator>
    template<typename Container>
    inline auto dynarray<T,Allocator>::insert(const_iterator pos,Container && container)noexcept
    -> type_<iterator,
        decltype(std::declval<Container>().begin()),
        decltype(std::declval<Container>().end()),
        decltype(std::declval<Container>().size())>
    {
        const difference_type offset = pos - begin();                   // using this to keep track (bookeep in case of iterator invalidation)
        const difference_type count = container.size();        // total number of elements to construct
        if ((size()+count)>capacity())                               
        {                                   
            grow_reserve(size() + count);        // invalidates iterators
            pos = begin() + offset;                     // using offset we can get back a iterator that points the correct element
        }
        try
        {
            const difference_type num_move_c =((end()-pos)<count) ? (end()-pos):count;   
            for (difference_type i = size_-1; i >=  size_-num_move_c; --i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i+count,std::move(data_buffer_[i]));
            } 
            auto container_last = container.end()-1;
            for (difference_type i = size_ + count - num_move_c-1;i >= size_; --i){
                std::allocator_traits<allocator_type>::construct(cap_alloc_.y(),data_buffer_+i,std::move(*(container_last)));
                --container_last;
            }       
            for (difference_type i =  size_-count-1; i >= offset; --i){       //loop for moving the other displaced elements
                data_buffer_[i+count] = std::move(data_buffer_[i]);
            }
            for (difference_type i = 0;i < num_move_c; ++i)
            {
                data_buffer_[offset+i] = *(container.begin()+i);
            }
            size_ += count;
        }
        catch(...)
        {
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
        return rand_access_iterator(pos.ptr());
    }
    
    template<typename T,typename Allocator>
    template<typename Container, typename index_container>
    inline auto dynarray<T,Allocator>::buffered_insert(const Container & insert_elements,index_container & insert_indices)
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(std::declval<index_container>().begin()),
            decltype(std::declval<index_container>().end()),
            decltype(std::declval<index_container>().size()),
            decltype(static_cast<int>(*(std::declval<index_container>().begin())))>
    {
        size_type count = std::distance(insert_elements.begin(),insert_elements.end());
        count = (count <=  std::distance(insert_indices.begin(),insert_indices.end())) ? count:std::distance(insert_indices.begin(),insert_indices.end());
        setup_buffered_insert(insert_indices,count);
        auto elit = insert_elements.begin();
        auto it = insert_indices.begin();
        for (size_type i = 0; i < count; ++i){
            data_buffer_[*it] = *elit;
            ++elit;
            ++it;
        }
    }
    
    template<typename T,typename Allocator>
    template<typename Container, typename indices>
    inline auto dynarray<T,Allocator>::buffered_insert(const Container & insert_elements,indices insert_indices[], size_type count)
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(static_cast<int>(std::declval<indices>()))>
    {
        setup_buffered_insert(insert_indices,count);
        auto elit = insert_elements.begin();
        for (size_type i = 0; i < count; ++i){
            data_buffer_[insert_indices[i]] = *elit;
            ++elit;
        }
    }

    template<typename T,typename Allocator>
    template<typename Container, typename index_container>
    inline auto dynarray<T,Allocator>::buffered_insert(Container && insert_elements,index_container & insert_indices)
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(std::declval<index_container>().begin()),
            decltype(std::declval<index_container>().end()),
            decltype(std::declval<index_container>().size()),
            decltype(static_cast<int>(*(std::declval<index_container>().begin())))>
    {
        size_type count = std::distance(insert_elements.begin(),insert_elements.end());
        count = (count <=  std::distance(insert_indices.begin(),insert_indices.end())) ? count:std::distance(insert_indices.begin(),insert_indices.end());
        setup_buffered_insert(insert_indices,count);
        auto elit = insert_elements.begin();
        auto it = insert_indices.begin();
        for (size_type i = 0; i < count; ++i){
            data_buffer_[*it] = std::move(*elit);
            ++elit;
            ++it;
        }
    }
    
    template<typename T,typename Allocator>
    template<typename Container, typename indices>
    inline auto dynarray<T,Allocator>::buffered_insert(Container && insert_elements,indices insert_indices[], size_type count)
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(static_cast<int>(std::declval<indices>()))>
    {
        setup_buffered_insert(insert_indices,count);
        auto elit = insert_elements.begin();
        for (size_type i = 0; i < count; ++i){
            data_buffer_[insert_indices[i]] = std::move(*elit);
            ++elit;
        }
    }


    template<typename T,typename Allocator>
    template<typename index_container>
    inline auto dynarray<T,Allocator>::buffered_erase(index_container & erase_indices)
        -> type_<void,
            decltype(std::declval<index_container>().begin()),
            decltype(std::declval<index_container>().end()),
            decltype(std::declval<index_container>().size()),
            decltype(static_cast<int>(*(std::declval<index_container>().begin())))>
    {
        size_type count = std::distance(erase_indices.begin(),erase_indices.end());
        const size_type old_size = size();
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<difference_type> d_allocator_type;
        d_allocator_type d_alloc(cap_alloc_.y());
        difference_type * arr_indx = reinterpret_cast<difference_type*>(std::allocator_traits<d_allocator_type>::allocate(d_alloc,old_size));
        for (size_type i = 0; i < size(); ++i){
            arr_indx[i] = i;
        }
        #pragma omp parallel for
        for (size_type j = 0; j < size(); ++j){
            for (const auto &i:erase_indices){
                arr_indx[j] = (arr_indx[j]==i) ? -1:arr_indx[j];
                arr_indx[j]-=static_cast<size_type>((arr_indx[j]>i));
            }
        }
        for (difference_type i = 0; i < size(); ++i){
            if (arr_indx[i] != -1)
            {
                data_buffer_[arr_indx[i]] = std::move(data_buffer_[i]);
            }
        }
        const size_type new_size = size() - count;
        resize(new_size);
        std::allocator_traits<d_allocator_type>::deallocate(d_alloc,reinterpret_cast<typename d_allocator_type::pointer>(arr_indx),old_size);
    }

    template<typename T,typename Allocator>
    template<typename indices>
    inline auto dynarray<T,Allocator>::buffered_erase(indices erase_indices[], size_type count)
        -> type_<void,
            decltype(static_cast<int>(std::declval<indices>()))>
    {
        const size_type old_size = size();
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<difference_type> d_allocator_type;
        d_allocator_type d_alloc(cap_alloc_.y());
        difference_type * arr_indx = reinterpret_cast<difference_type*>(std::allocator_traits<d_allocator_type>::allocate(d_alloc,old_size));
        for (size_type i = 0; i < size(); ++i){
            arr_indx[i] = i;
        }
        #pragma omp parallel for
        for (size_type j = 0; j < size(); ++j){
            for (size_type i = 0 ; i < count; ++i){
                arr_indx[j] = (arr_indx[j]==erase_indices[i]) ? -1:arr_indx[j];
                arr_indx[j]-=static_cast<size_type>((arr_indx[j]>erase_indices[i]));
            }
        }
        for (difference_type i = 0; i < size(); ++i){
            if (arr_indx[i] != -1)
            {
                data_buffer_[arr_indx[i]] = std::move(data_buffer_[i]);
            }
        }
        const size_type new_size = size() - count;
        resize(new_size);
        std::allocator_traits<d_allocator_type>::deallocate(d_alloc,reinterpret_cast<typename d_allocator_type::pointer>(arr_indx),old_size);
    }

    template<typename T,typename Allocator>
    inline dynarray<T,Allocator>::iterator dynarray<T,Allocator>::erase(const_iterator pos)noexcept{
        const difference_type offset = pos - begin();
        for (difference_type i = offset; i < size_-1; ++i){
            data_buffer_[i] = std::move(data_buffer_[i+1]);
        }
        std::allocator_traits<allocator_type>::destroy(cap_alloc_.y(),data_buffer_ + size_-1);
        size_ -=1;
        return iterator(data_buffer_+offset);
    }

    template<typename T,typename Allocator>
    inline dynarray<T,Allocator>::iterator dynarray<T,Allocator>::erase(const_iterator first, const_iterator last)noexcept{
        const difference_type offset = first - begin();
        const difference_type count = last - first;
        for (difference_type i = offset; i < size_ - count; ++i){
            data_buffer_[i] = std::move(data_buffer_[i+count]);
        }
        destroy_elements(size_ - count,size_);
        return iterator(data_buffer_+offset);
    }

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::push_back(const_reference value)noexcept{
        append(std::forward<const T&>(value));
    }

    template<typename T,typename Allocator>
    inline void dynarray<T,Allocator>::push_back(T&& value)noexcept{
        append(std::forward<T&&>(value));
    }

    template<typename T,typename Allocator>
    template<typename... Args>
    inline dynarray<T,Allocator>::iterator dynarray<T,Allocator>::emplace(const_iterator pos, Args&&...args)noexcept{
        return insert_one(pos,std::forward<Args>(args)...);
    }

    template<typename T,typename Allocator>
    template<typename... Args>
    inline dynarray<T,Allocator>::reference dynarray<T,Allocator>::emplace_back(Args && ...args)noexcept{
        append(std::forward<Args>(args)...);
        return data_buffer_[size()-1];
    }

    template<typename T, typename Allocator>
    inline void dynarray<T,Allocator>::pop_back()noexcept{
        // will crash and burn if the dynarray is empty obviously
        try{
            std::allocator_traits<allocator_type>::destroy(cap_alloc_.y(),&data_buffer_[size()-1]);
            --size_;
        }catch(...){
            std::cerr << "Failed to call pop_back on hopeless dynarray" << '\n';
            std::exception_ptr exception=std::current_exception();
            cout_exception(exception);
        }
    }

    template<typename T, typename Allocator>
    inline void dynarray<T,Allocator>::resize(size_type new_size)noexcept{
        resize_arr(new_size);
    }

    template<typename T, typename Allocator>
    inline void dynarray<T,Allocator>::resize(size_type new_size, const_reference value)noexcept{
        resize_arr(new_size,std::forward<const_reference>(value));
    }
#endif
}
#endif