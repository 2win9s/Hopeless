// Implementation for a 2d ragged array (r2darray) implemented with elements stored 
// as one hopeless::dynarray and indexing(using spans with dynamic extent) in another
// this is to reduce memory fragmentation compared to a vector of vectors
// also can be access on openmp offload device with operator ()   
// e.g. r3darray(i)(j) or r2darray(i,j) 
#pragma once
#ifndef HOPELESS_RAGGED_ARRAY
#define HOPELESS_RAGGED_ARRAY

#include<iostream>
#include<utility>
#include<cstddef>
#include<numeric>
#include<algorithm>
#include<omp.h>

#include"dynarray.hpp"
#include"dyn_extent_span.hpp"
#include"hopeless_macros_n_meta.hpp"


namespace hopeless{
#ifdef HOPELESS_TARGET_OMP_DEV
    template<typename T,typename Allocator = hopeless::allocator<T>,
                        int dev_no = HOPELESS_DEFAULT_OMP_OFFLOAD_DEV>
    struct r2darray;

    template<typename InputIt>
    constexpr inline auto count_elements(const InputIt first, const InputIt last)noexcept 
        -> type_<std::ptrdiff_t,
            decltype((*first).size()),
            decltype(first != last)
            >
    {
        std::ptrdiff_t sum = 0;
        for (auto i = first; i != last; ++i){
            sum += (*i).size();
        }
        return sum;
    }

    template<typename T,typename Allocator,int dev_no>     
    auto inline operator <<(std::ostream& stream,
        const r2darray<T,Allocator,dev_no> & r2darray) 
            -> type_<std::ostream&, 
            decltype(std::cout<<std::declval<T>())>;  

    template<typename T,typename Allocator,int dev_no>
    void swap(r2darray<T,Allocator,dev_no>& r2darray1, r2darray<T,Allocator,dev_no>& r2darray2)noexcept;

    template<typename T,typename Allocator,int dev_no>     
    struct r2darray{

        static_assert(std::is_trivially_copyable_v<T>, "type should be trivially copyable");
        static_assert(std::is_copy_constructible_v<T>, "type should be trivially copy constructible");
        static_assert(std::is_copy_assignable_v<T>, "type should be trivially copy assignable");

        struct rand_access_iterator;
        struct const_rand_access_iterator;
        typedef T value_type;
        typedef Allocator allocator_type;
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<dyn_extent_span<T>> dspan_alloctor_type;
        typedef typename dspan_alloctor_type::pointer dspan_alloc_ptr;
        typedef std::ptrdiff_t difference_type;
        typedef std::ptrdiff_t size_type;       //I'm sick of the unsigned to signed conversion warning with arithmetic nonsense I'm on a 64bit system when are you going to use an array so big the signed one would fail?
        typedef value_type& reference;
        typedef const value_type& const_reference;
        typedef typename std::allocator_traits<Allocator>::pointer pointer;	
        typedef typename std::allocator_traits<Allocator>::const_pointer const_pointer;	
        typedef rand_access_iterator iterator;
        typedef const_rand_access_iterator const_iterator;

        typedef typename std::reverse_iterator<iterator> reverse_iterator;
        typedef typename std::reverse_iterator<const_iterator> const_reverse_iterator;

        constexpr inline dyn_extent_span<T> operator [](const size_type i)const noexcept;

        #pragma omp declare target device_type(nohost)
        constexpr inline dyn_extent_span<T> operator ()(const size_type i)const noexcept;
        constexpr inline reference operator ()(const size_type i, const size_type j)const noexcept;
        constexpr inline reference operator ()(const size_type i, const size_type j)noexcept;
        #pragma omp end declare target

        struct rand_access_iterator  
        {
        public:
            typedef std::random_access_iterator_tag iterator_category;   
            typedef dyn_extent_span<T> value_type;
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

        struct const_rand_access_iterator : public rand_access_iterator  
        {
        public:
            typedef std::random_access_iterator_tag iterator_category;   
            typedef dyn_extent_span<T> value_type;
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
            inline const_pointer operator ->()const noexcept {return (this->iter);}
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

        void swap(r2darray& rhs)noexcept;


        constexpr r2darray()noexcept(noexcept(Allocator()));
        constexpr explicit r2darray(const Allocator& alloc)noexcept;
        
        r2darray( const r2darray& other )noexcept;
        r2darray( const r2darray& other, const Allocator& alloc )noexcept;
        r2darray( r2darray&& other)noexcept;
        r2darray( r2darray&& other, const Allocator& alloc)noexcept;

        r2darray( std::initializer_list<size_type> init,const Allocator& alloc = Allocator())noexcept;
        r2darray( std::initializer_list<std::initializer_list<T>> init,const Allocator& alloc = Allocator())noexcept;

        r2darray(const_iterator first,const_iterator last,const Allocator& alloc = Allocator())noexcept;
                
        ~r2darray()noexcept;
        
        template<int dev_no2>
        r2darray& operator =(const r2darray<T,Allocator,dev_no2> & other)noexcept;

        r2darray& operator =(const r2darray & other)noexcept;
        r2darray& operator =(r2darray && other)noexcept;
    
    protected:      // functions used in the constructors to reduce code duplication
        inline void create_indexing_buffer()noexcept;
        inline void create_dev_indexing_buffer()noexcept;
        void construct_span_indexing(const dyn_extent_span<T> otheridx[])noexcept;
        void construct_span_indexing(std::initializer_list<size_type>& init)noexcept;
        void construct_span_indexing(std::initializer_list<std::initializer_list<T>> & init)noexcept;
        void construct_span_indexing(const_iterator first)noexcept;
    
    public:
        // reset the pointers of the spans in indexing after invalidation
        void reset_indexing_spans()noexcept;
        constexpr inline allocator_type get_allocator()const noexcept;
        constexpr inline size_type size()const noexcept;
        constexpr inline size_type capacity()const noexcept;

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

        inline void resize(size_type new_size)noexcept;             // change the number of rows
        inline void resize_row(size_type row,size_type new_size, const T & fill = T())noexcept;         // change the number of rows
        inline void resize_row(iterator row,size_type new_size, const T & fill = T())noexcept;         // change the number of rows
        inline void erase_row(size_type row)noexcept;
        inline void erase_row(iterator row)noexcept;
        template<typename Container>
        inline auto insert_row(size_type row, const Container & container)    
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size())>;

        template<typename Container>
        inline auto insert_row(iterator row, const Container & container)    
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size())>;

        template<typename Container, typename index_container>
        inline auto buffered_insert(const Container & insert_elements,const  index_container & row_indices,const index_container & column_indices)            
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size()),
                decltype(std::declval<index_container>().begin()),
                decltype(std::declval<index_container>().end()),
                decltype(std::declval<index_container>().size()),
                decltype(static_cast<int>(*(std::declval<index_container>().begin())))>;
        
        template<typename index_container>
        inline auto buffered_erase(const index_container & row_indices,const index_container & column_indices)            
            -> type_<void,
                decltype(std::declval<index_container>().begin()),
                decltype(std::declval<index_container>().end()),
                decltype(std::declval<index_container>().size()),
                decltype(static_cast<int>(*(std::declval<index_container>().begin())))>;
        
        inline void map_to_omp_dev();
        inline void map_from_omp_dev();
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
    protected:
        dynarray<T,Allocator,dev_no> data_vec_;
        dyn_extent_span<T>* indexing_vec_;
        size_type size_;
        packed_pair<size_type,allocator_type> cap_alloc_;
        dyn_extent_span<T>* dev_indexing_vec_; // the copy on the default offloading device
    };

    template<typename T,typename Allocator,int dev_no>     
    auto inline operator <<(std::ostream& stream,
        const r2darray<T,Allocator,dev_no> & r2darray) 
            -> type_<std::ostream&, 
                decltype(std::cout<<std::declval<T>())>
    {
        stream << "[";
                                // size is a signed type
        for(intptr_t i = 0; i < r2darray.size()-1; ++i){
            stream << r2darray[i] << ",\n";
        }
        if (r2darray.size()){
            stream << r2darray[r2darray.size()-1];
        }
        stream << "]";
        return stream;
    }


    template<typename T,typename Allocator,int dev_no>
    void swap(r2darray<T,Allocator,dev_no>& r2darray1, r2darray<T,Allocator,dev_no>& r2darray2)noexcept{
        r2darray1.swap(r2darray2);
    }

    template<typename T,typename Allocator,int dev_no>    
    constexpr inline dyn_extent_span<T> r2darray<T,Allocator,dev_no>::operator [](const size_type i)const noexcept{
        return indexing_vec_[i];
    }

    #pragma omp declare target device_type(nohost)
    template<typename T,typename Allocator,int dev_no>    
    constexpr inline dyn_extent_span<T> r2darray<T,Allocator,dev_no>::operator ()(const size_type i)const noexcept{
        return dev_indexing_vec_[i];
    }
    template<typename T,typename Allocator,int dev_no>    
    constexpr inline r2darray<T,Allocator,dev_no>::reference r2darray<T,Allocator,dev_no>::operator ()(const size_type i, const size_type j)const noexcept{
        return dev_indexing_vec_[i][j];
    }
    template<typename T,typename Allocator,int dev_no>    
    constexpr inline r2darray<T,Allocator,dev_no>::reference r2darray<T,Allocator,dev_no>::operator ()(const size_type i, const size_type j)noexcept{
        return dev_indexing_vec_[i][j];
    }
    #pragma omp end declare target
    
    template<typename T,typename Allocator,int dev_no>
    void r2darray<T,Allocator,dev_no>::swap(r2darray & rhs)noexcept{
        using std::swap;
        swap(this->data_vec_,rhs.data_vec_);
        swap(this->indexing_vec_,rhs.indexing_vec_);
        swap(this->size_,rhs.size_);
        swap(this->cap_alloc_,rhs.cap_alloc_);
        swap(this->dev_indexing_vec_,rhs.dev_indexing_vec_);
    }

    template<typename T,typename Allocator,int dev_no>    
    constexpr r2darray<T,Allocator,dev_no>::r2darray()noexcept(noexcept(Allocator()))        
        :data_vec_(Allocator()),
        indexing_vec_(nullptr),
        size_(0),
        cap_alloc_(0,Allocator()),
        dev_indexing_vec_(nullptr)
    {}
    
    template<typename T,typename Allocator,int dev_no>    
    constexpr r2darray<T,Allocator,dev_no>::r2darray(const Allocator& alloc)noexcept
        :data_vec_(alloc),
        indexing_vec_(nullptr),
        size_(0),
        cap_alloc_(0,alloc),
        dev_indexing_vec_(nullptr)
    {}
    
    template<typename T,typename Allocator,int dev_no>    
    r2darray<T,Allocator,dev_no>::r2darray(const r2darray& other)noexcept
        :data_vec_(other.data_vec_,std::allocator_traits<allocator_type>::select_on_container_copy_construction(
        other.get_allocator())),
        indexing_vec_(nullptr),
        size_(other.size()),
        cap_alloc_(other.size(),std::allocator_traits<allocator_type>::select_on_container_copy_construction(
        other.get_allocator())),
        dev_indexing_vec_(nullptr)
    {
        create_indexing_buffer();
        construct_span_indexing(other.indexing_vec_);
    }

    template<typename T,typename Allocator,int dev_no>    
    r2darray<T,Allocator,dev_no>::r2darray(const r2darray& other, const Allocator& alloc)noexcept
        :data_vec_(other.data_vec_,alloc),
        indexing_vec_(other.indexing_vec_),
        size_(other.size()),
        cap_alloc_(other.size(),alloc),
        dev_indexing_vec_(other.dev_indexing_vec_)
    {
        create_indexing_buffer();
        construct_span_indexing(other.indexing_vec_);
    }

    template<typename T,typename Allocator,int dev_no>    
    r2darray<T,Allocator,dev_no>::r2darray(r2darray&& other)noexcept
        :data_vec_(),
        indexing_vec_(nullptr),
        size_(0),
        cap_alloc_(0,other.get_allocator()),
        dev_indexing_vec_(nullptr)
    {
        using std::swap;
        swap(*this,std::forward<r2darray&>(other));
    }

    template<typename T,typename Allocator,int dev_no>    
    r2darray<T,Allocator,dev_no>::r2darray(r2darray&& other,const Allocator& alloc)noexcept
        :data_vec_(std::forward<dynarray<T,Allocator,dev_no>&&>(other.data_vec_),alloc),
        indexing_vec_(other.indexing_vec_),
        size_(other.size()),
        cap_alloc_(other.size(),alloc),
        dev_indexing_vec_(other.dev_indexing_vec_)
    {
        create_indexing_buffer();
        construct_span_indexing(other.indexing_vec_);
    }

    template<typename T,typename Allocator,int dev_no>    
    r2darray<T,Allocator,dev_no>::r2darray(std::initializer_list<size_type> init,const Allocator& alloc)noexcept
        :data_vec_(std::reduce(init.begin(),init.end(),0),alloc),
        indexing_vec_(nullptr),
        size_(init.size()),
        cap_alloc_(init.size(),alloc),
        dev_indexing_vec_(nullptr)
    {
        create_indexing_buffer();
        construct_span_indexing(std::forward<std::initializer_list<size_type>&>(init));
    }

    template<typename T,typename Allocator,int dev_no>    
    r2darray<T,Allocator,dev_no>::r2darray(std::initializer_list<std::initializer_list<T>> init,const Allocator& alloc)noexcept
        :data_vec_(alloc),
        indexing_vec_(nullptr),
        size_(init.size()),
        cap_alloc_(init.size(),alloc),
        dev_indexing_vec_(nullptr)
    {
        data_vec_.reserve(count_elements(init.begin(),init.end()));
        for (const auto i : init){
            for (const auto j : i){
                data_vec_.emplace_back(j);
            }
        }
        create_indexing_buffer();
        construct_span_indexing(std::forward<std::initializer_list<std::initializer_list<T>>&>(init));
    }

    template<typename T,typename Allocator,int dev_no>    
    r2darray<T,Allocator,dev_no>::r2darray(const_iterator first,const_iterator last,const Allocator& alloc)noexcept
        :data_vec_(alloc),
        indexing_vec_(nullptr),
        size_(last - first),
        cap_alloc_(last - first,alloc),
        dev_indexing_vec_(nullptr)
    {
        data_vec_.reserve(count_elements(first,last));
        for (auto i = first; i != last; ++i){
            for (const auto j : *i){
                data_vec_.emplace_back(j);
            }
        }
        create_indexing_buffer();
        construct_span_indexing(first);
    }

    template<typename T,typename Allocator,int dev_no>    
    r2darray<T,Allocator,dev_no>::~r2darray()noexcept
    {
        try{
            dspan_alloctor_type dspan_alloc(cap_alloc_.y());
            std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<dspan_alloc_ptr>(indexing_vec_),capacity());
        }
        catch(...){
            std::cerr << "r2darray failed to deallocate memory" << '\n';
        }
        omp_target_free(dev_indexing_vec_,dev_no);
    }

    template<typename T,typename Allocator,int dev_no> 
    template<int dev_no2>   
    r2darray<T,Allocator,dev_no>& r2darray<T,Allocator,dev_no>::operator=(const r2darray<T,Allocator,dev_no2> & other)noexcept
    {
        data_vec_ = other.data_vec_;
        size_ = other.size();
        if (other.size() > capacity())
        {
            try{
                dspan_alloctor_type dspan_alloc(cap_alloc_.y());
                std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<pointer>(indexing_vec_),capacity());
                omp_target_free(dev_indexing_vec_,dev_no);
                cap_alloc_.x() = other.size();
                create_indexing_buffer();
            }
            catch(...){
                std::cerr << "r2darray failed to deallocate memory" << '\n';
                size_ = 0;
            }
        }
        construct_span_indexing(other.indexing_vec_);   // spans do not need destructor calls
        return *this;
    }

    template<typename T,typename Allocator,int dev_no> 
    r2darray<T,Allocator,dev_no>& r2darray<T,Allocator,dev_no>::operator=(const r2darray & other)noexcept
    {
        data_vec_ = other.data_vec_;
        size_ = other.size();
        if (other.size() > capacity())
        {
            try{
                dspan_alloctor_type dspan_alloc(cap_alloc_.y());
                std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<pointer>(indexing_vec_),capacity());
                omp_target_free(dev_indexing_vec_,dev_no);
                cap_alloc_.x() = other.size();
                create_indexing_buffer();
            }
            catch(...){
                std::cerr << "r2darray failed to deallocate memory" << '\n';
                size_ = 0;
            }
        }
        construct_span_indexing(other.indexing_vec_);   // spans do not need destructor calls
        return *this;
    }

    template<typename T,typename Allocator,int dev_no> 
    r2darray<T,Allocator,dev_no>& r2darray<T,Allocator,dev_no>::operator=(r2darray && other)noexcept
    {
        using std::swap;
        swap(*this, std::forward<r2darray&>(other));
        return *this;
    }

    template<typename T,typename Allocator,int dev_no>    
    inline void r2darray<T,Allocator,dev_no>::create_indexing_buffer()noexcept{
        dspan_alloctor_type dspan_alloc(cap_alloc_.y());
        auto temp = std::allocator_traits<dspan_alloctor_type>::allocate(dspan_alloc,capacity());
        if (temp){
            indexing_vec_ = reinterpret_cast<dyn_extent_span<T>*>(temp);
            create_dev_indexing_buffer();
        }else{
            if (size())
            {
                std::cerr<<"ERROR r2darray creation failed to allocate memory"<<std::endl;
                size_ = 0;
            }
        }
    }

    template<typename T,typename Allocator,int dev_no>    
    void r2darray<T,Allocator,dev_no>::construct_span_indexing(const dyn_extent_span<T> otheridx[])noexcept{
        if (size())
        {
            dspan_alloctor_type dspan_alloc(cap_alloc_.y());
            dyn_extent_span<T> * temp = reinterpret_cast<dyn_extent_span<T>*>(std::allocator_traits<dspan_alloctor_type>::allocate(dspan_alloc,size()));
            const auto start = data_vec_.data();
            const auto dstart = data_vec_.data_dev(); 
            std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,indexing_vec_,start,otheridx[0].size());
            std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp[0],dstart,otheridx[0].size());
            #pragma omp parallel for           
            for (size_type i = 1; i < size(); ++i){
                const difference_type ptr_offset = (otheridx[i].data()-otheridx[0].data());
                std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&indexing_vec_[i],
                                    start + ptr_offset, otheridx[i].size());
                std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp[i],
                                    dstart + ptr_offset, otheridx[i].size());
            }
            try{
                bool fail = omp_target_memcpy(dev_indexing_vec_,&temp[0],sizeof(dyn_extent_span<T>) * size(),0,0,dev_no,omp_get_initial_device());
                if(fail){
                    throw std::runtime_error("ERROR r2darray failed to copy data to device memory");
                }
            }catch(std::runtime_error& e){
                std::cerr<<e.what()<<std::endl;
            }catch(...){
                std::cerr<<"Unexpected error, possible memory corruption?"<<std::endl;
            }
            std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<dspan_alloc_ptr>(temp),size());
        }
    }

    template<typename T,typename Allocator,int dev_no>    
    void r2darray<T,Allocator,dev_no>::construct_span_indexing(std::initializer_list<size_type>& init)noexcept{
        if (size())
        {
            dspan_alloctor_type dspan_alloc(cap_alloc_.y());
            dyn_extent_span<T> * temp = reinterpret_cast<dyn_extent_span<T>*>(std::allocator_traits<dspan_alloctor_type>::allocate(dspan_alloc,size()));
            typedef typename std::allocator_traits<allocator_type>::rebind_alloc<size_type> s_allocator_type;
            s_allocator_type s_alloc(cap_alloc_.y());
            size_type * temp_offsets = reinterpret_cast<size_type*>(std::allocator_traits<s_allocator_type>::allocate(s_alloc,size()));
            const auto start = data_vec_.data();
            const auto dstart = data_vec_.data_dev(); 
            auto init_iter = init.begin();
            std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,indexing_vec_,start,*init_iter);
            std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp[0],dstart,*init_iter);
            temp_offsets[0] = 0;
            for (size_type i = 0; i < size()-1; ++i){
                temp_offsets[i+1] = temp_offsets[i]+(*(init_iter + i));
            }
            #pragma omp parallel for           
            for (size_type i = 1; i < size(); ++i){
                std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&indexing_vec_[i],
                                    start + temp_offsets[i], *(init_iter + i));  //an intialiser list iterator is just a pointer so this arithmetic is fine
                std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp[i],
                                    dstart + temp_offsets[i],*(init_iter + i));
            }
            try{
                bool fail = omp_target_memcpy(dev_indexing_vec_,&temp[0],sizeof(dyn_extent_span<T>) * size(),0,0,dev_no,omp_get_initial_device());
                if(fail){
                    throw std::runtime_error("ERROR r2darray failed to copy data to device memory");
                }
            }catch(std::runtime_error& e){
                std::cerr<<e.what()<<std::endl;
            }catch(...){
                std::cerr<<"Unexpected error, possible memory corruption?"<<std::endl;
            }
            std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<dspan_alloc_ptr>(temp),size());
            std::allocator_traits<s_allocator_type>::deallocate(s_alloc,reinterpret_cast<typename s_allocator_type::pointer>(temp_offsets),size());
        }
    }

    template<typename T,typename Allocator,int dev_no>    
    void r2darray<T,Allocator,dev_no>::construct_span_indexing(std::initializer_list<std::initializer_list<T>> & init)noexcept{
        if (size())
        {
            dspan_alloctor_type dspan_alloc(cap_alloc_.y());
            dyn_extent_span<T> * temp = reinterpret_cast<dyn_extent_span<T>*>(std::allocator_traits<dspan_alloctor_type>::allocate(dspan_alloc,size()));
            typedef typename std::allocator_traits<allocator_type>::rebind_alloc<size_type> s_allocator_type;
            s_allocator_type s_alloc(cap_alloc_.y());
            size_type * temp_offsets = reinterpret_cast<size_type*>(std::allocator_traits<s_allocator_type>::allocate(s_alloc,size()));
            const auto start = data_vec_.data();
            const auto dstart = data_vec_.data_dev(); 
            auto init_iter = init.begin();
            std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,indexing_vec_,start,(*init_iter).size());
            std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp[0],dstart,(*init_iter).size());
            temp_offsets[0] = 0;
            for (size_type i = 0; i < size()-1; ++i){
                temp_offsets[i+1] = temp_offsets[i]+(*(init_iter + i)).size();
            }
            #pragma omp parallel for           
            for (size_type i = 1; i < size(); ++i){
                std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&indexing_vec_[i],
                                    start + temp_offsets[i], (*(init_iter + i)).size());  //an intialiser list iterator is just a pointer so this arithmetic is fine
                std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp[i],
                                    dstart + temp_offsets[i],(*(init_iter + i)).size());
            }
            try{
                bool fail = omp_target_memcpy(dev_indexing_vec_,&temp[0],sizeof(dyn_extent_span<T>) * size(),0,0,dev_no,omp_get_initial_device());
                if(fail){
                    throw std::runtime_error("ERROR r2darray failed to copy data to device memory");
                }
            }catch(std::runtime_error& e){
                std::cerr<<e.what()<<std::endl;
            }catch(...){
                std::cerr<<"Unexpected error, possible memory corruption?"<<std::endl;
            }
            std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<dspan_alloc_ptr>(temp),size());
            std::allocator_traits<s_allocator_type>::deallocate(s_alloc,reinterpret_cast<typename s_allocator_type::pointer>(temp_offsets),size());
        }
    }

    template<typename T,typename Allocator,int dev_no>    
    void r2darray<T,Allocator,dev_no>::construct_span_indexing(const_iterator first)noexcept{
        if (size())
        {
            dspan_alloctor_type dspan_alloc(cap_alloc_.y());
            dyn_extent_span<T> * temp = reinterpret_cast<dyn_extent_span<T>*>(std::allocator_traits<dspan_alloctor_type>::allocate(dspan_alloc,size()));
            const auto start = data_vec_.data();
            const auto dstart = data_vec_.data_dev(); 
            std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,indexing_vec_,start,(*first).size());
            std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp[0],dstart,(*first).size());
            const auto other_ptr_begin = (*first).data();
            #pragma omp parallel for           
            for (size_type i = 1; i < size(); ++i){
                const difference_type ptr_offset = ((*(first+i)).data()-other_ptr_begin);
                const size_type row_size = (*(first + i)).size();
                std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&indexing_vec_[i],
                                    start + ptr_offset, row_size);  
                std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp[i],
                                    dstart + ptr_offset, row_size);
            }
            try{
                bool fail = omp_target_memcpy(dev_indexing_vec_,&temp[0],sizeof(dyn_extent_span<T>) * size(),0,0,dev_no,omp_get_initial_device());
                if(fail){
                    throw std::runtime_error("ERROR r2darray failed to copy data to device memory");
                }
            }catch(std::runtime_error& e){
                std::cerr<<e.what()<<std::endl;
            }catch(...){
                std::cerr<<"Unexpected error, possible memory corruption?"<<std::endl;
            }
            std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<dspan_alloc_ptr>(temp),size());
        }
    }

    template<typename T,typename Allocator,int dev_no>    
    inline void r2darray<T,Allocator,dev_no>::create_dev_indexing_buffer()noexcept{
        auto temp =(dyn_extent_span<T>*)omp_target_alloc(capacity()*sizeof(dyn_extent_span<T>),dev_no);
        if (temp){
            dev_indexing_vec_ = temp;
        }else if(size()>0){
            std::cerr<<"ERROR r2darray creation failed to allocate memory on offload device,\ndo not define TARGET_OMP_DEV macro for dynarray if not offloading with openmp"
            <<",\n ensure the device has enough memory available"<<std::endl;
        }
    }

    template<typename T,typename Allocator,int dev_no>    
    void r2darray<T,Allocator,dev_no>::reset_indexing_spans()noexcept{
        if (((bool)(size()))){
            const difference_type offset = data_vec_.data()-indexing_vec_[0].data();  
            indexing_vec_[0].change_span_ptr(data_vec_.data());
            #pragma omp target device(dev_no)
            {
                const difference_type dev_offset = data_vec_.data_dev()-dev_indexing_vec_[0].data();  
                dev_indexing_vec_[0].change_span_ptr(data_vec_.data_dev());
                #pragma omp loop
                for (size_type i = 1; i < size(); ++i)
                {
                    dev_indexing_vec_[i].change_span_ptr(dev_indexing_vec_[i].data()+dev_offset);
                }
            }
            #pragma omp parallel for
            for (size_type i = 1; i < size(); ++i)
            {
                indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()+offset);
            }
        }
    }

    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::allocator_type r2darray<T,Allocator,dev_no>::get_allocator()const noexcept{
        return cap_alloc_.y();
    }
    
    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::size_type r2darray<T,Allocator,dev_no>::size()const noexcept{
        return size_;
    }

    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::size_type r2darray<T,Allocator,dev_no>::capacity()const noexcept{
        return cap_alloc_.x();
    }
    
    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::iterator r2darray<T,Allocator,dev_no>::begin()noexcept{
        return iterator(indexing_vec_);
    }
    
    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::const_iterator r2darray<T,Allocator,dev_no>::begin()const noexcept{
        return const_iterator(indexing_vec_);
    }
    
    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::const_iterator r2darray<T,Allocator,dev_no>::cbegin()const noexcept{
        return const_iterator(indexing_vec_);
    }

    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::iterator r2darray<T,Allocator,dev_no>::end()noexcept{
        return iterator(indexing_vec_+size());
    }
    
    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::const_iterator r2darray<T,Allocator,dev_no>::end()const noexcept{
        return const_iterator(indexing_vec_+size());
    }
    
    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::const_iterator r2darray<T,Allocator,dev_no>::cend()const noexcept{
        return const_iterator(indexing_vec_+size());
    }


    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::reverse_iterator r2darray<T,Allocator,dev_no>::rbegin()noexcept{
        return reverse_iterator(iterator(indexing_vec_));
    }
    
    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::const_reverse_iterator r2darray<T,Allocator,dev_no>::rbegin()const noexcept{
        return const_reverse_iterator(const_iterator(indexing_vec_));
    }
    
    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::const_reverse_iterator r2darray<T,Allocator,dev_no>::crbegin()const noexcept{
        return const_reverse_iterator(const_iterator(indexing_vec_));
    }

    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::reverse_iterator r2darray<T,Allocator,dev_no>::rend()noexcept{
        return reverse_iterator(iterator(indexing_vec_+size()));
    }
    
    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::const_reverse_iterator r2darray<T,Allocator,dev_no>::rend()const noexcept{
        return const_reverse_iterator(const_iterator(indexing_vec_+size()));
    }
    
    template<typename T,typename Allocator,int dev_no> 
    constexpr inline r2darray<T,Allocator,dev_no>::const_reverse_iterator r2darray<T,Allocator,dev_no>::crend()const noexcept{
        return const_reverse_iterator(const_iterator(indexing_vec_+size()));
    }

    template<typename T,typename Allocator,int dev_no> 
    inline void r2darray<T,Allocator,dev_no>::resize(size_type new_size)noexcept{
        if (new_size > capacity()){

            dspan_alloctor_type dspan_alloc(cap_alloc_.y());
            dyn_extent_span<T>* temp = reinterpret_cast<dyn_extent_span<T>*>(std::allocator_traits<dspan_alloctor_type>::allocate(dspan_alloc,new_size));
            if (temp){
                const size_type new_rows = new_size - size();
                #pragma omp parallel for
                for (size_type i = 0; i < size(); ++i){
                    std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp[i],std::move(indexing_vec_[i]));
                }
                dyn_extent_span<T> * temp_spans = reinterpret_cast<dyn_extent_span<T>*>(std::allocator_traits<dspan_alloctor_type>::allocate(dspan_alloc,new_rows));
                const auto end = indexing_vec_[size()-1].data() + indexing_vec_[size()-1].size();  //one past the end pointer
                const auto devend = end - data_vec_.data() + data_vec_.data_dev();                
                #pragma omp parallel for
                for (size_type i = size(); i < new_size; ++i){
                    std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp[i],
                                        end,0);  
                    std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp_spans[i-size()],
                                        devend,0);
                }
                try{
                    std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<dspan_alloc_ptr>(indexing_vec_),capacity());
                    indexing_vec_ = reinterpret_cast<dyn_extent_span<T>*>(temp);
                }
                catch(...)
                {
                    std::cerr<<"ERROR r2darray resize failed to allocate memory"<<std::endl;
                    std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<dspan_alloc_ptr>(temp),new_size);
                }
                auto temp2 =(dyn_extent_span<T>*)omp_target_alloc(new_size*sizeof(dyn_extent_span<T>),dev_no);
                if (temp2){
                    omp_target_memcpy(temp2,dev_indexing_vec_,sizeof(dyn_extent_span<T>) * size(),0,0,dev_no,dev_no);         
                    omp_target_memcpy(temp2,temp_spans,sizeof(dyn_extent_span<T>) * new_rows,sizeof(dyn_extent_span<T>) * size(),0,dev_no,omp_get_initial_device());
                    omp_target_free(dev_indexing_vec_,dev_no);
                    dev_indexing_vec_ = temp2;
                }else{
                    std::cerr<<"ERROR r2darray resize failed to allocate memory on offload device,\ndo not define TARGET_OMP_DEV macro for dynarray if not offloading with openmp"
                    <<",\n ensure the device has enough memory available"<<std::endl;
                }
                std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<dspan_alloc_ptr>(temp_spans),new_size);
                size_ = new_size;
                cap_alloc_.x() = new_size;
            }else{
                std::cerr<<"ERROR r2darray resize failed to allocate memory"<<std::endl;
                size_ = 0;
            }
        }else{
            data_vec_.resize(count_elements(begin(),begin()+new_size));
            size_ = new_size;
        }
    }

    template<typename T,typename Allocator,int dev_no> 
    inline void r2darray<T,Allocator,dev_no>::resize_row(size_type row,size_type new_size,const T & fill)noexcept{
        if (new_size == indexing_vec_[row].size()){
            return;
        }
        if (new_size > indexing_vec_[row].size()){
            const difference_type new_elements_count = new_size - indexing_vec_[row].size();
            if (data_vec_.capacity() < data_vec_.size() + new_elements_count){
                data_vec_.grow_reserve_no_map(data_vec_.size() + new_elements_count);     // any pointer invalidation happens here
                reset_indexing_spans();
            }  
            #pragma omp target map(to:row,new_size,new_elements_count) device(dev_no)
            {
                dev_indexing_vec_[row].resize(new_size);
                #pragma omp loop
                for (size_type i = row+1; i < size(); ++i){
                    dev_indexing_vec_[i].change_span_ptr(dev_indexing_vec_[i].data()+new_elements_count);
                }
            }
            typename dynarray<T,Allocator,dev_no>::rand_access_iterator pos(indexing_vec_[row].data()+indexing_vec_[row].size()); 
            data_vec_.insert(pos,new_elements_count,fill);           
            indexing_vec_[row].resize(new_size);
            #pragma omp parallel for
            for (size_type i = row+1; i < size(); ++i){
                indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()+new_elements_count);
            }
        }else{
            const difference_type erase_elements_count = indexing_vec_[row].size() - new_size;
            #pragma omp target map(to:new_size,erase_elements_count) device(dev_no)
            {
                dev_indexing_vec_[row].resize(new_size);
                #pragma omp loop
                for (size_type i = row+1; i < size(); ++i){
                    dev_indexing_vec_[i].change_span_ptr(dev_indexing_vec_[i].data()-erase_elements_count);
                }
            }
            typename dynarray<T,Allocator,dev_no>::rand_access_iterator pos(indexing_vec_[row].data()+indexing_vec_[row].size());
            data_vec_.erase(pos - erase_elements_count,pos);
            indexing_vec_[row].resize(new_size);
            #pragma omp parallel for
            for (size_type i = row+1; i < size(); ++i){
                indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()-erase_elements_count);
            }
        }
    }

    template<typename T,typename Allocator,int dev_no> 
    inline void r2darray<T,Allocator,dev_no>::resize_row(iterator row,size_type new_size,const T & fill)noexcept{
        if (new_size == row->size()){
            return;
        } 
        if (new_size > row->size()){
            const difference_type new_elements_count = new_size - row->size();
            const difference_type row_pos = row - begin();    
            if (data_vec_.capacity() < data_vec_.size() + new_elements_count){
                data_vec_.grow_reserve_no_map(data_vec_.size() + new_elements_count);     // any pointer invalidation happens here
                reset_indexing_spans();
            }
            #pragma omp target map(to:row_pos,new_elements_count) device(dev_no)
            {
                dev_indexing_vec_[row_pos].resize(new_size);
                #pragma omp loop
                for (size_type i = row_pos+1; i < size(); ++i){
                    dev_indexing_vec_[i].change_span_ptr(dev_indexing_vec_[i].data()+new_elements_count);
                }
            }
            typename dynarray<T,Allocator,dev_no>::rand_access_iterator pos(row->data()+row->size()); 
            data_vec_.insert(pos,new_elements_count,fill);           
            row->resize(new_size);

            #pragma omp parallel for
            for (size_type i = row_pos+1; i < size(); ++i){
                indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()+new_elements_count);
            }
        }else{
            const difference_type erase_elements_count = row->size() - new_size;
            const difference_type row_pos = row - begin();
            #pragma omp target map(to:new_size,row_pos,erase_elements_count) device(dev_no)
            {
                dev_indexing_vec_[row_pos].resize(new_size);
                #pragma omp loop
                for (size_type i = row_pos+1; i < size(); ++i){
                    dev_indexing_vec_[i].change_span_ptr(dev_indexing_vec_[i].data()-erase_elements_count);
                }
            }
            typename dynarray<T,Allocator,dev_no>::rand_access_iterator pos(row->data()+row->size());
            data_vec_.erase(pos - erase_elements_count,pos);
            row->resize(new_size);
            #pragma omp parallel for
            for (size_type i = row_pos+1; i < size(); ++i){
                indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()-erase_elements_count);
            }
        }
    }
    
    template<typename T,typename Allocator,int dev_no> 
    inline void r2darray<T,Allocator,dev_no>::erase_row(size_type row)noexcept{
        #pragma omp target map(to:row) device(dev_no)
        {
            #pragma omp loop
            for (size_type i = row+1; i < size(); ++i){
                dev_indexing_vec_[i].change_span_ptr(dev_indexing_vec_[i].data()-dev_indexing_vec_[row].size());
            }
            for (size_type i = row+1; i < size(); ++i){
                dev_indexing_vec_[i-1] = dev_indexing_vec_[i];
            }
        }
        typename dynarray<T,Allocator,dev_no>::rand_access_iterator pos(indexing_vec_[row].data());
        data_vec_.erase(pos,pos + indexing_vec_[row].size());
        #pragma omp parallel for
        for (size_type i = row+1; i < size(); ++i){
            indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()-indexing_vec_[row].size());
        }
        for (size_type i = row+1; i < size(); ++i){
            indexing_vec_[i-1] = std::move(indexing_vec_[i]);
        }
        size_-=1;
    }

    template<typename T,typename Allocator,int dev_no> 
    inline void r2darray<T,Allocator,dev_no>::erase_row(iterator row)noexcept{
        const difference_type row_pos = row - begin();
        #pragma omp target map(to:row_pos) device(dev_no)
        {
            #pragma omp loop
            for (size_type i = row_pos+1; i < size(); ++i){
                dev_indexing_vec_[i].change_span_ptr(dev_indexing_vec_[i].data()-dev_indexing_vec_[row_pos].size());
            }
            for (auto i = row_pos+1; i < size(); ++i){
                dev_indexing_vec_[i-1] = dev_indexing_vec_[i];
            }
        }
        typename dynarray<T,Allocator,dev_no>::rand_access_iterator pos(row->data());
        data_vec_.erase(pos,pos + row->size());
        #pragma omp parallel for
        for (size_type i = row_pos+1; i < size(); ++i){
            indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()-row->size());
        }
        for (size_type i = row_pos+1; i < size(); ++i){
            indexing_vec_[i-1] = std::move(indexing_vec_[i]);
        }
        size_-=1;
    }

    template<typename T,typename Allocator,int dev_no> 
    template<typename Container>
    inline auto r2darray<T,Allocator,dev_no>::insert_row(size_type row, const Container & container)    
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())>
    {
        const size_type new_elements_count = container.size();
        resize(size() + 1);
        if (data_vec_.capacity() < data_vec_.size() + new_elements_count){
            data_vec_.grow_reserve_no_map(data_vec_.size() + new_elements_count);     // any pointer invalidation happens here
            reset_indexing_spans();
        }
        typename dynarray<T,Allocator,dev_no>::rand_access_iterator pos(indexing_vec_[row].data());
        data_vec_.insert(pos, container.begin(),container.end());
        #pragma omp target map(to:row,new_elements_count)
        {
            for (size_type i = size() - 1; i > row; --i){
                dev_indexing_vec_[i] = dev_indexing_vec_[i-1];
            }
            dev_indexing_vec_[row].resize(new_elements_count);
            #pragma omp loop
            for (size_type i = row+1; i < size(); ++i)
            {
                dev_indexing_vec_[i].change_span_ptr(dev_indexing_vec_[i].data() + new_elements_count);
            }
        }
        for (size_type i =  size() - 1; i > row; --i){
            indexing_vec_[i] = indexing_vec_[i-1];
        }
        indexing_vec_[row].resize(new_elements_count);
        #pragma omp parallel for
        for (size_type i = row+1; i < size(); ++i)
        {
            indexing_vec_[i].change_span_ptr(indexing_vec_[i].data() + new_elements_count);
        }
    }

    template<typename T,typename Allocator,int dev_no> 
    template<typename Container>
    inline auto r2darray<T,Allocator,dev_no>::insert_row(iterator row, const Container & container)    
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())>
    {
        const difference_type row_pos = row - begin();
        const size_type new_elements_count = container.size();
        resize(size() + 1);
        if (data_vec_.capacity() < data_vec_.size() + new_elements_count){
            data_vec_.grow_reserve_no_map(data_vec_.size() + new_elements_count);     // any pointer invalidation happens here
            reset_indexing_spans();
        }
        typename dynarray<T,Allocator,dev_no>::rand_access_iterator pos(indexing_vec_[row_pos].data());
        data_vec_.insert(pos, container.begin(),container.end());
        #pragma omp target map(to:row_pos,new_elements_count)
        {
            for (size_type i = size() - 1; i > row_pos; --i){
                dev_indexing_vec_[i] = dev_indexing_vec_[i-1];
            }
            dev_indexing_vec_[row_pos].resize(new_elements_count);
            #pragma omp loop
            for (size_type i = row_pos+1; i < size(); ++i)
            {
                dev_indexing_vec_[i].change_span_ptr(dev_indexing_vec_[i].data() + new_elements_count);
            }
        }
        for (size_type i =  size() - 1; i > row_pos; --i){
            indexing_vec_[i] = indexing_vec_[i-1];
        }
        indexing_vec_[row_pos].resize(new_elements_count);
        #pragma omp parallel for
        for (size_type i = row_pos+1; i < size(); ++i)
        {
            indexing_vec_[i].change_span_ptr(indexing_vec_[i].data() + new_elements_count);
        }
    }
    

    template<typename T,typename Allocator,int dev_no> 
    template<typename Container, typename index_container>
    inline auto r2darray<T,Allocator,dev_no>::buffered_insert(const Container & insert_elements,const  index_container & row_indices,const index_container & column_indices)     
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(std::declval<index_container>().begin()),
            decltype(std::declval<index_container>().end()),
            decltype(std::declval<index_container>().size()),
            decltype(static_cast<int>(*(std::declval<index_container>().begin())))>
    {
        const size_type new_elements_count = std::min({insert_elements.size(),row_indices.size(),column_indices.size()}); // minimum of the size of the 3 input containers
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<size_type> s_allocator_type;
        s_allocator_type s_alloc(cap_alloc_.y());
        size_type * insert_data_vec_idx = reinterpret_cast<size_type*>(std::allocator_traits<s_allocator_type>::allocate(s_alloc,new_elements_count));         // for storing equivalent index for the flat dynarray
        if (data_vec_.capacity() < data_vec_.size() + new_elements_count){
            data_vec_.grow_reserve_no_map(data_vec_.size() + new_elements_count);     // any pointer invalidation happens here
            const difference_type offset = data_vec_.data()-indexing_vec_[0].data();  
            indexing_vec_[0].change_span_ptr(data_vec_.data());
            #pragma omp parallel for
            for (size_type i = 1; i < size(); ++i){
                indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()+offset);
            }
        }
        auto row_it = row_indices.begin();
        auto col_it = column_indices.begin();
        for (size_type i = 0; i < new_elements_count; ++i)
        {
            insert_data_vec_idx[i] = indexing_vec_[*row_it].data() - data_vec_.data() + *col_it;
            #pragma omp parallel for
            for (size_type j = *row_it + 1; j < size(); ++j)
            {
                indexing_vec_[j].change_span_ptr(indexing_vec_[j].data() + 1);
            }
            indexing_vec_[*row_it].resize(indexing_vec_[*row_it].size() + 1);
            ++row_it;
            ++col_it;
        }
        
        omp_target_memcpy(dev_indexing_vec_,indexing_vec_,sizeof(dyn_extent_span<T>) * size(),0 ,0,dev_no,omp_get_initial_device());
        #pragma omp target device(dev_no)
        {
            const difference_type dev_offset = data_vec_.data_dev()-dev_indexing_vec_[0].data();  
            dev_indexing_vec_[0].change_span_ptr(data_vec_.data_dev());
            #pragma omp loop
            for (size_type i = 1; i < size(); ++i)
            {
                dev_indexing_vec_[i].change_span_ptr(dev_indexing_vec_[i].data()+dev_offset);
            }
        }
        data_vec_.buffered_insert(insert_elements,insert_data_vec_idx,new_elements_count);
        std::allocator_traits<s_allocator_type>::deallocate(s_alloc,reinterpret_cast<typename s_allocator_type::pointer>(insert_data_vec_idx),new_elements_count);
    }

    template<typename T,typename Allocator,int dev_no> 
    template<typename index_container>
    inline auto r2darray<T,Allocator,dev_no>::buffered_erase(const index_container & row_indices,const index_container & column_indices)            
        -> type_<void,
            decltype(std::declval<index_container>().begin()),
            decltype(std::declval<index_container>().end()),
            decltype(std::declval<index_container>().size()),
            decltype(static_cast<int>(*(std::declval<index_container>().begin())))>
    {
        const size_type erase_elements_count = (row_indices.size() > column_indices.size()) ? row_indices.size():column_indices.size();
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<size_type> s_allocator_type;
        s_allocator_type s_alloc(cap_alloc_.y());
        size_type * erase_data_vec_idx = reinterpret_cast<size_type*>(std::allocator_traits<s_allocator_type>::allocate(s_alloc,erase_elements_count));         // for storing equivalent index for the flat dynarray
        auto row_it = row_indices.begin();
        auto col_it = column_indices.begin();
        for (size_type i = 0; i < erase_elements_count; ++i)
        {
            erase_data_vec_idx[i] = indexing_vec_[*row_it].data() - data_vec_.data() + *col_it;

            #pragma omp parallel for
            for (size_type j = *row_it + 1; j < size(); ++j)
            {
                indexing_vec_[j].change_span_ptr(indexing_vec_[j].data() - 1);
            }
            indexing_vec_[*row_it].resize(indexing_vec_[*row_it].size() - 1);
            ++row_it;
            ++col_it;
        }
        omp_target_memcpy(dev_indexing_vec_,indexing_vec_,sizeof(dyn_extent_span<T>) * size(),0 ,0,dev_no,omp_get_initial_device());
        #pragma omp target device(dev_no)
        {
            const difference_type dev_offset = data_vec_.data_dev()-dev_indexing_vec_[0].data();  
            dev_indexing_vec_[0].change_span_ptr(data_vec_.data_dev());
            #pragma omp loop
            for (size_type i = 1; i < size(); ++i)
            {
                dev_indexing_vec_[i].change_span_ptr(dev_indexing_vec_[i].data()+dev_offset);
            }
        }
        data_vec_.buffered_erase(erase_data_vec_idx,erase_elements_count);
        std::allocator_traits<s_allocator_type>::deallocate(s_alloc,reinterpret_cast<typename s_allocator_type::pointer>(erase_data_vec_idx),erase_elements_count);    
    }

    template<typename T,typename Allocator,int dev_no> 
    inline void r2darray<T,Allocator,dev_no>::map_to_omp_dev(){
        data_vec_.map_data_to_omp_dev();
    }

    template<typename T,typename Allocator,int dev_no> 
    inline void r2darray<T,Allocator,dev_no>::map_from_omp_dev(){
        data_vec_.map_data_from_omp_dev();
    }
    /*
    // I may never implement this, indexing multidimensional jagged arrays is quite inefficient
    template<typename T,typename Allocator,int dev_no>     
    struct rndarray{
        dynarray<T,Allocator,dev_no> data_vec_;
        r2darray<dyn_extent_span<T>,Allocator,dev_no> indexing_vec_;
    };*/
#else
    template<typename T,typename Allocator = hopeless::allocator<T>>
    struct r2darray;

    template<typename InputIt>
    constexpr inline auto count_elements(const InputIt first, const InputIt last)noexcept 
        -> type_<std::ptrdiff_t,
            decltype((*first).size()),
            decltype(first != last)
            >
    {
        std::ptrdiff_t sum = 0;
        for (auto i = first; i != last; ++i){
            sum += (*i).size();
        }
        return sum;
    }

    template<typename T,typename Allocator>     
    auto inline operator <<(std::ostream& stream,
        const r2darray<T,Allocator> & r2darray) 
            -> type_<std::ostream&, 
            decltype(std::cout<<std::declval<T>())>;  

    template<typename T,typename Allocator>
    void swap(r2darray<T,Allocator>& r2darray1, r2darray<T,Allocator>& r2darray2)noexcept;

    template<typename T,typename Allocator>     
    struct r2darray{
        struct rand_access_iterator;
        struct const_rand_access_iterator;
        typedef T value_type;
        typedef Allocator allocator_type;
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<dyn_extent_span<T>> dspan_alloctor_type;
        typedef typename dspan_alloctor_type::pointer dspan_alloc_ptr;
        typedef std::ptrdiff_t difference_type;
        typedef std::ptrdiff_t size_type;       //I'm sick of the unsigned to signed conversion warning with arithmetic nonsense I'm on a 64bit system when are you going to use an array so big the signed one would fail?
        typedef value_type& reference;
        typedef const value_type& const_reference;
        typedef typename std::allocator_traits<Allocator>::pointer pointer;	
        typedef typename std::allocator_traits<Allocator>::const_pointer const_pointer;	
        typedef rand_access_iterator iterator;
        typedef const_rand_access_iterator const_iterator;

        typedef typename std::reverse_iterator<iterator> reverse_iterator;
        typedef typename std::reverse_iterator<const_iterator> const_reverse_iterator;

        constexpr inline dyn_extent_span<T> operator [](const size_type i)const noexcept;
        constexpr inline dyn_extent_span<T> operator ()(const size_type i)const noexcept;
        constexpr inline reference operator ()(const size_type i, const size_type j)const noexcept;
        constexpr inline reference operator ()(const size_type i, const size_type j)noexcept;

        struct rand_access_iterator  
        {
        public:
            typedef std::random_access_iterator_tag iterator_category;   
            typedef dyn_extent_span<T> value_type;
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

        struct const_rand_access_iterator : public rand_access_iterator  
        {
        public:
            typedef std::random_access_iterator_tag iterator_category;   
            typedef dyn_extent_span<T> value_type;
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
            inline const_pointer operator ->()const noexcept {return (this->iter);}
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

        void swap(r2darray& rhs)noexcept;


        constexpr r2darray()noexcept(noexcept(Allocator()));
        constexpr explicit r2darray(const Allocator& alloc)noexcept;
        
        r2darray( const r2darray& other )noexcept;
        r2darray( const r2darray& other, const Allocator& alloc )noexcept;
        r2darray( r2darray&& other)noexcept;
        r2darray( r2darray&& other, const Allocator& alloc)noexcept;

        r2darray( std::initializer_list<size_type> init,const Allocator& alloc = Allocator())noexcept;
        r2darray( std::initializer_list<std::initializer_list<T>> init,const Allocator& alloc = Allocator())noexcept;

        r2darray(const_iterator first,const_iterator last,const Allocator& alloc = Allocator())noexcept;
                
        ~r2darray()noexcept;

        r2darray& operator =(const r2darray & other)noexcept;
        r2darray& operator =(r2darray && other)noexcept;
    
    protected:      // functions used in the constructors to reduce code duplication
        inline void create_indexing_buffer()noexcept;
        void construct_span_indexing(const dyn_extent_span<T> otheridx[])noexcept;
        void construct_span_indexing(std::initializer_list<size_type>& init)noexcept;
        void construct_span_indexing(std::initializer_list<std::initializer_list<T>> & init)noexcept;
        void construct_span_indexing(const_iterator first)noexcept;
    
    public:
        // reset the pointers of the spans in indexing after invalidation
        void reset_indexing_spans()noexcept;
        constexpr inline allocator_type get_allocator()const noexcept;
        constexpr inline size_type size()const noexcept;
        constexpr inline size_type capacity()const noexcept;

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

        inline void resize(size_type new_size)noexcept;             // change the number of rows
        inline void resize_row(size_type row,size_type new_size, const T & fill = T())noexcept;         // change the number of rows
        inline void resize_row(iterator row,size_type new_size, const T & fill = T())noexcept;         // change the number of rows
        inline void erase_row(size_type row)noexcept;
        inline void erase_row(iterator row)noexcept;
        template<typename Container>
        inline auto insert_row(size_type row, const Container & container)    
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size())>;

        template<typename Container>
        inline auto insert_row(iterator row, const Container & container)    
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size())>;

        template<typename Container, typename index_container>
        inline auto buffered_insert(const Container & insert_elements,const  index_container & row_indices,const index_container & column_indices)            
            -> type_<void,
                decltype(std::declval<Container>().begin()),
                decltype(std::declval<Container>().end()),
                decltype(std::declval<Container>().size()),
                decltype(std::declval<index_container>().begin()),
                decltype(std::declval<index_container>().end()),
                decltype(std::declval<index_container>().size()),
                decltype(static_cast<int>(*(std::declval<index_container>().begin())))>;
        
        template<typename index_container>
        inline auto buffered_erase(const index_container & row_indices,const index_container & column_indices)            
            -> type_<void,
                decltype(std::declval<index_container>().begin()),
                decltype(std::declval<index_container>().end()),
                decltype(std::declval<index_container>().size()),
                decltype(static_cast<int>(*(std::declval<index_container>().begin())))>;
    
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
    protected:
        dynarray<T,Allocator> data_vec_;
        dyn_extent_span<T>* indexing_vec_;
        size_type size_;
        packed_pair<size_type,allocator_type> cap_alloc_;
    };

    template<typename T,typename Allocator>     
    auto inline operator <<(std::ostream& stream,
        const r2darray<T,Allocator> & r2darray) 
            -> type_<std::ostream&, 
                decltype(std::cout<<std::declval<T>())>
    {
        stream << "[";
                                // size is a signed type
        for(intptr_t i = 0; i < r2darray.size()-1; ++i){
            stream << r2darray[i] << ",\n";
        }
        if (r2darray.size()){
            stream << r2darray[r2darray.size()-1];
        }
        stream << "]";
        return stream;
    }


    template<typename T,typename Allocator>
    void swap(r2darray<T,Allocator>& r2darray1, r2darray<T,Allocator>& r2darray2)noexcept{
        r2darray1.swap(r2darray2);
    }

    template<typename T,typename Allocator>    
    constexpr inline dyn_extent_span<T> r2darray<T,Allocator>::operator [](const size_type i)const noexcept{
        return indexing_vec_[i];
    }

    template<typename T,typename Allocator>    
    constexpr inline dyn_extent_span<T> r2darray<T,Allocator>::operator ()(const size_type i)const noexcept{
        return indexing_vec_[i];
    }
    template<typename T,typename Allocator>    
    constexpr inline r2darray<T,Allocator>::reference r2darray<T,Allocator>::operator ()(const size_type i, const size_type j)const noexcept{
        return indexing_vec_[i][j];
    }
    template<typename T,typename Allocator>    
    constexpr inline r2darray<T,Allocator>::reference r2darray<T,Allocator>::operator ()(const size_type i, const size_type j)noexcept{
        return indexing_vec_[i][j];
    }

    
    template<typename T,typename Allocator>
    void r2darray<T,Allocator>::swap(r2darray & rhs)noexcept{
        using std::swap;
        swap(this->data_vec_,rhs.data_vec_);
        swap(this->indexing_vec_,rhs.indexing_vec_);
        swap(this->size_,rhs.size_);
        swap(this->cap_alloc_,rhs.cap_alloc_);
    }

    template<typename T,typename Allocator>    
    constexpr r2darray<T,Allocator>::r2darray()noexcept(noexcept(Allocator()))        
        :data_vec_(Allocator()),
        indexing_vec_(nullptr),
        size_(0),
        cap_alloc_(0,Allocator())
    {}
    
    template<typename T,typename Allocator>    
    constexpr r2darray<T,Allocator>::r2darray(const Allocator& alloc)noexcept
        :data_vec_(alloc),
        indexing_vec_(nullptr),
        size_(0),
        cap_alloc_(0,alloc)
    {}
    
    template<typename T,typename Allocator>    
    r2darray<T,Allocator>::r2darray(const r2darray& other)noexcept
        :data_vec_(other.data_vec_,std::allocator_traits<allocator_type>::select_on_container_copy_construction(
        other.get_allocator())),
        indexing_vec_(nullptr),
        size_(other.size()),
        cap_alloc_(other.size(),std::allocator_traits<allocator_type>::select_on_container_copy_construction(
        other.get_allocator()))
    {
        create_indexing_buffer();
        construct_span_indexing(other.indexing_vec_);
    }

    template<typename T,typename Allocator>    
    r2darray<T,Allocator>::r2darray(const r2darray& other, const Allocator& alloc)noexcept
        :data_vec_(other.data_vec_,alloc),
        indexing_vec_(other.indexing_vec_),
        size_(other.size()),
        cap_alloc_(other.size(),alloc)
    {
        create_indexing_buffer();
        construct_span_indexing(other.indexing_vec_);
    }

    template<typename T,typename Allocator>    
    r2darray<T,Allocator>::r2darray(r2darray&& other)noexcept
        :data_vec_(),
        indexing_vec_(nullptr),
        size_(0),
        cap_alloc_(0,other.get_allocator())
    {
        using std::swap;
        swap(*this,std::forward<r2darray&>(other));
    }

    template<typename T,typename Allocator>    
    r2darray<T,Allocator>::r2darray(r2darray&& other,const Allocator& alloc)noexcept
        :data_vec_(std::forward<dynarray<T,Allocator>&&>(other.data_vec_),alloc),
        indexing_vec_(other.indexing_vec_),
        size_(other.size()),
        cap_alloc_(other.size(),alloc)
    {
        create_indexing_buffer();
        construct_span_indexing(other.indexing_vec_);
    }

    template<typename T,typename Allocator>    
    r2darray<T,Allocator>::r2darray(std::initializer_list<size_type> init,const Allocator& alloc)noexcept
        :data_vec_(std::reduce(init.begin(),init.end(),0),alloc),
        indexing_vec_(nullptr),
        size_(init.size()),
        cap_alloc_(init.size(),alloc)
    {
        create_indexing_buffer();
        construct_span_indexing(std::forward<std::initializer_list<size_type>&>(init));
    }

    template<typename T,typename Allocator>    
    r2darray<T,Allocator>::r2darray(std::initializer_list<std::initializer_list<T>> init,const Allocator& alloc)noexcept
        :data_vec_(alloc),
        indexing_vec_(nullptr),
        size_(init.size()),
        cap_alloc_(init.size(),alloc)
    {
        data_vec_.reserve(count_elements(init.begin(),init.end()));
        for (const auto i : init){
            for (const auto j : i){
                data_vec_.emplace_back(j);
            }
        }
        create_indexing_buffer();
        construct_span_indexing(std::forward<std::initializer_list<std::initializer_list<T>>&>(init));
    }

    template<typename T,typename Allocator>    
    r2darray<T,Allocator>::r2darray(const_iterator first,const_iterator last,const Allocator& alloc)noexcept
        :data_vec_(alloc),
        indexing_vec_(nullptr),
        size_(last - first),
        cap_alloc_(last - first,alloc)
    {
        data_vec_.reserve(count_elements(first,last));
        for (auto i = first; i != last; ++i){
            for (const auto j : *i){
                data_vec_.emplace_back(j);
            }
        }
        create_indexing_buffer();
        construct_span_indexing(first);
    }

    template<typename T,typename Allocator>    
    r2darray<T,Allocator>::~r2darray()noexcept
    {
        try{
            dspan_alloctor_type dspan_alloc(cap_alloc_.y());
            std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<dspan_alloc_ptr>(indexing_vec_),capacity());
        }
        catch(...){
            std::cerr << "r2darray failed to deallocate memory" << '\n';
        }
    }

    template<typename T,typename Allocator> 
    r2darray<T,Allocator>& r2darray<T,Allocator>::operator=(const r2darray & other)noexcept
    {
        data_vec_ = other.data_vec_;
        size_ = other.size();
        if (other.size() > capacity())
        {
            try{
                dspan_alloctor_type dspan_alloc(cap_alloc_.y());
                std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<pointer>(indexing_vec_),capacity());
                cap_alloc_.x() = other.size();
                create_indexing_buffer();
            }
            catch(...){
                std::cerr << "r2darray failed to deallocate memory" << '\n';
                size_ = 0;
            }
        }
        construct_span_indexing(other.indexing_vec_);   // spans do not need destructor calls
        return *this;
    }

    template<typename T,typename Allocator> 
    r2darray<T,Allocator>& r2darray<T,Allocator>::operator=(r2darray && other)noexcept
    {
        using std::swap;
        swap(*this, std::forward<r2darray&>(other));
        return *this;
    }

    template<typename T,typename Allocator>    
    inline void r2darray<T,Allocator>::create_indexing_buffer()noexcept{
        dspan_alloctor_type dspan_alloc(cap_alloc_.y());
        auto temp = std::allocator_traits<dspan_alloctor_type>::allocate(dspan_alloc,capacity());
        if (temp){
            indexing_vec_ = reinterpret_cast<dyn_extent_span<T>*>(temp);
        }else{
            if (size())
            {
                std::cerr<<"ERROR r2darray creation failed to allocate memory"<<std::endl;
                size_ = 0;
            }
        }
    }

    template<typename T,typename Allocator>    
    void r2darray<T,Allocator>::construct_span_indexing(const dyn_extent_span<T> otheridx[])noexcept{
        if (size())
        {
            dspan_alloctor_type dspan_alloc(cap_alloc_.y());
            const auto start = data_vec_.data();
            std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,indexing_vec_,start,otheridx[0].size());
            #pragma omp parallel for           
            for (size_type i = 1; i < size(); ++i){
                const difference_type ptr_offset = (otheridx[i].data()-otheridx[0].data());
                std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&indexing_vec_[i],
                                    start + ptr_offset, otheridx[i].size());
            }
        }
    }

    template<typename T,typename Allocator>    
    void r2darray<T,Allocator>::construct_span_indexing(std::initializer_list<size_type>& init)noexcept{
        if (size())
        {
            dspan_alloctor_type dspan_alloc(cap_alloc_.y());
            const auto start = data_vec_.data();
            auto init_iter = init.begin();
            std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,indexing_vec_,start,*init_iter);
            for (size_type i = 1; i < size(); ++i){
                std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&indexing_vec_[i],
                            indexing_vec_[i-1].data() + indexing_vec_[i-1].size(), *(init_iter + i));  //an intialiser list iterator is just a pointer so this arithmetic is fine
            }
        }
    }

    template<typename T,typename Allocator>    
    void r2darray<T,Allocator>::construct_span_indexing(std::initializer_list<std::initializer_list<T>> & init)noexcept{
        if (size())
        {
            dspan_alloctor_type dspan_alloc(cap_alloc_.y());
            const auto start = data_vec_.data();
            auto init_iter = init.begin();
            std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,indexing_vec_,start,(*init_iter).size());
            for (size_type i = 1; i < size(); ++i){
                std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&indexing_vec_[i],
                            indexing_vec_[i-1].data() + indexing_vec_[i-1].size(), (*(init_iter + i)).size());  //an intialiser list iterator is just a pointer so this arithmetic is fine
            }
        }
    }

    template<typename T,typename Allocator>    
    void r2darray<T,Allocator>::construct_span_indexing(const_iterator first)noexcept{
        if (size())
        {
            dspan_alloctor_type dspan_alloc(cap_alloc_.y());
            const auto start = data_vec_.data();
            std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,indexing_vec_,start,(*first).size());
            const auto other_ptr_begin = (*first).data();
            #pragma omp parallel for           
            for (size_type i = 1; i < size(); ++i){
                const difference_type ptr_offset = ((*(first+i)).data()-other_ptr_begin);
                const size_type row_size = (*(first + i)).size();
                std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&indexing_vec_[i],
                                    start + ptr_offset, row_size);  
            }
        }
    }

    template<typename T,typename Allocator>    
    void r2darray<T,Allocator>::reset_indexing_spans()noexcept{
        if (((bool)(size()))){
            const difference_type offset = data_vec_.data()-indexing_vec_[0].data();  
            indexing_vec_[0].change_span_ptr(data_vec_.data());
            #pragma omp parallel for
            for (size_type i = 1; i < size(); ++i)
            {
                indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()+offset);
            }
        }
    }

    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::allocator_type r2darray<T,Allocator>::get_allocator()const noexcept{
        return cap_alloc_.y();
    }
    
    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::size_type r2darray<T,Allocator>::size()const noexcept{
        return size_;
    }

    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::size_type r2darray<T,Allocator>::capacity()const noexcept{
        return cap_alloc_.x();
    }
    
    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::iterator r2darray<T,Allocator>::begin()noexcept{
        return iterator(indexing_vec_);
    }
    
    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::const_iterator r2darray<T,Allocator>::begin()const noexcept{
        return const_iterator(indexing_vec_);
    }
    
    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::const_iterator r2darray<T,Allocator>::cbegin()const noexcept{
        return const_iterator(indexing_vec_);
    }

    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::iterator r2darray<T,Allocator>::end()noexcept{
        return iterator(indexing_vec_+size());
    }
    
    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::const_iterator r2darray<T,Allocator>::end()const noexcept{
        return const_iterator(indexing_vec_+size());
    }
    
    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::const_iterator r2darray<T,Allocator>::cend()const noexcept{
        return const_iterator(indexing_vec_+size());
    }


    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::reverse_iterator r2darray<T,Allocator>::rbegin()noexcept{
        return reverse_iterator(iterator(indexing_vec_));
    }
    
    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::const_reverse_iterator r2darray<T,Allocator>::rbegin()const noexcept{
        return const_reverse_iterator(const_iterator(indexing_vec_));
    }
    
    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::const_reverse_iterator r2darray<T,Allocator>::crbegin()const noexcept{
        return const_reverse_iterator(const_iterator(indexing_vec_));
    }

    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::reverse_iterator r2darray<T,Allocator>::rend()noexcept{
        return reverse_iterator(iterator(indexing_vec_+size()));
    }
    
    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::const_reverse_iterator r2darray<T,Allocator>::rend()const noexcept{
        return const_reverse_iterator(const_iterator(indexing_vec_+size()));
    }
    
    template<typename T,typename Allocator> 
    constexpr inline r2darray<T,Allocator>::const_reverse_iterator r2darray<T,Allocator>::crend()const noexcept{
        return const_reverse_iterator(const_iterator(indexing_vec_+size()));
    }

    template<typename T,typename Allocator> 
    inline void r2darray<T,Allocator>::resize(size_type new_size)noexcept{
        if (new_size > capacity()){

            dspan_alloctor_type dspan_alloc(cap_alloc_.y());
            dyn_extent_span<T>* temp = reinterpret_cast<dyn_extent_span<T>*>(std::allocator_traits<dspan_alloctor_type>::allocate(dspan_alloc,new_size));
            if (temp){
                const size_type new_rows = new_size - size();
                #pragma omp parallel for
                for (size_type i = 0; i < size(); ++i){
                    std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp[i],std::move(indexing_vec_[i]));
                }
                const auto end = indexing_vec_[size()-1].data() + indexing_vec_[size()-1].size();  //one past the end pointer              
                #pragma omp parallel for
                for (size_type i = size(); i < new_size; ++i){
                    std::allocator_traits<dspan_alloctor_type>::construct(dspan_alloc,&temp[i],
                                        end,0);  
                }
                try{
                    std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<dspan_alloc_ptr>(indexing_vec_),capacity());
                    indexing_vec_ = reinterpret_cast<dyn_extent_span<T>*>(temp);
                }
                catch(...)
                {
                    std::cerr<<"ERROR r2darray resize failed to allocate memory"<<std::endl;
                    std::allocator_traits<dspan_alloctor_type>::deallocate(dspan_alloc,reinterpret_cast<dspan_alloc_ptr>(temp),new_size);
                }
                size_ = new_size;
                cap_alloc_.x() = new_size;
            }else{
                std::cerr<<"ERROR r2darray resize failed to allocate memory"<<std::endl;
                size_ = 0;
            }
        }else{
            data_vec_.resize(count_elements(begin(),begin()+new_size));
            size_ = new_size;
        }
    }

    template<typename T,typename Allocator> 
    inline void r2darray<T,Allocator>::resize_row(size_type row,size_type new_size,const T & fill)noexcept{
        if (new_size == indexing_vec_[row].size()){
            return;
        }
        if (new_size > indexing_vec_[row].size()){
            const difference_type new_elements_count = new_size - indexing_vec_[row].size();
            if (data_vec_.capacity() < data_vec_.size() + new_elements_count){
                data_vec_.grow_reserve(data_vec_.size() + new_elements_count);     // any pointer invalidation happens here
                reset_indexing_spans();
            }  
            typename dynarray<T,Allocator>::rand_access_iterator pos(indexing_vec_[row].data()+indexing_vec_[row].size()); 
            data_vec_.insert(pos,new_elements_count,fill);           
            indexing_vec_[row].resize(new_size);
            #pragma omp parallel for
            for (size_type i = row+1; i < size(); ++i){
                indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()+new_elements_count);
            }
        }else{
            const difference_type erase_elements_count = indexing_vec_[row].size() - new_size;
            typename dynarray<T,Allocator>::rand_access_iterator pos(indexing_vec_[row].data()+indexing_vec_[row].size());
            data_vec_.erase(pos - erase_elements_count,pos);
            indexing_vec_[row].resize(new_size);
            #pragma omp parallel for
            for (size_type i = row+1; i < size(); ++i){
                indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()-erase_elements_count);
            }
        }
    }

    template<typename T,typename Allocator> 
    inline void r2darray<T,Allocator>::resize_row(iterator row,size_type new_size,const T & fill)noexcept{
        if (new_size == row->size()){
            return;
        } 
        if (new_size > row->size()){
            const difference_type new_elements_count = new_size - row->size();
            const difference_type row_pos = row - begin();    
            if (data_vec_.capacity() < data_vec_.size() + new_elements_count){
                data_vec_.grow_reserve(data_vec_.size() + new_elements_count);     // any pointer invalidation happens here
                reset_indexing_spans();
            }
            typename dynarray<T,Allocator>::rand_access_iterator pos(row->data()+row->size()); 
            data_vec_.insert(pos,new_elements_count,fill);           
            row->resize(new_size);

            #pragma omp parallel for
            for (size_type i = row_pos+1; i < size(); ++i){
                indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()+new_elements_count);
            }
        }else{
            const difference_type erase_elements_count = row->size() - new_size;
            const difference_type row_pos = row - begin();
            typename dynarray<T,Allocator>::rand_access_iterator pos(row->data()+row->size());
            data_vec_.erase(pos - erase_elements_count,pos);
            row->resize(new_size);
            #pragma omp parallel for
            for (size_type i = row_pos+1; i < size(); ++i){
                indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()-erase_elements_count);
            }
        }
    }
    
    template<typename T,typename Allocator> 
    inline void r2darray<T,Allocator>::erase_row(size_type row)noexcept{
        typename dynarray<T,Allocator>::rand_access_iterator pos(indexing_vec_[row].data());
        data_vec_.erase(pos,pos + indexing_vec_[row].size());
        #pragma omp parallel for
        for (size_type i = row+1; i < size(); ++i){
            indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()-indexing_vec_[row].size());
        }
        for (size_type i = row+1; i < size(); ++i){
            indexing_vec_[i-1] = std::move(indexing_vec_[i]);
        }
        size_-=1;
    }

    template<typename T,typename Allocator> 
    inline void r2darray<T,Allocator>::erase_row(iterator row)noexcept{
        const difference_type row_pos = row - begin();
        typename dynarray<T,Allocator>::rand_access_iterator pos(row->data());
        data_vec_.erase(pos,pos + row->size());
        #pragma omp parallel for
        for (size_type i = row_pos+1; i < size(); ++i){
            indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()-row->size());
        }
        for (size_type i = row_pos+1; i < size(); ++i){
            indexing_vec_[i-1] = std::move(indexing_vec_[i]);
        }
        size_-=1;
    }

    template<typename T,typename Allocator> 
    template<typename Container>
    inline auto r2darray<T,Allocator>::insert_row(size_type row, const Container & container)    
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())>
    {
        const size_type new_elements_count = container.size();
        resize(size() + 1);
        if (data_vec_.capacity() < data_vec_.size() + new_elements_count){
            data_vec_.grow_reserve(data_vec_.size() + new_elements_count);     // any pointer invalidation happens here
            reset_indexing_spans();
        }
        typename dynarray<T,Allocator>::rand_access_iterator pos(indexing_vec_[row].data());
        data_vec_.insert(pos, container.begin(),container.end());
        for (size_type i =  size() - 1; i > row; --i){
            indexing_vec_[i] = indexing_vec_[i-1];
        }
        indexing_vec_[row].resize(new_elements_count);
        #pragma omp parallel for
        for (size_type i = row+1; i < size(); ++i)
        {
            indexing_vec_[i].change_span_ptr(indexing_vec_[i].data() + new_elements_count);
        }
    }

    template<typename T,typename Allocator> 
    template<typename Container>
    inline auto r2darray<T,Allocator>::insert_row(iterator row, const Container & container)    
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size())>
    {
        const difference_type row_pos = row - begin();
        const size_type new_elements_count = container.size();
        resize(size() + 1);
        if (data_vec_.capacity() < data_vec_.size() + new_elements_count){
            data_vec_.grow_reserve(data_vec_.size() + new_elements_count);     // any pointer invalidation happens here
            reset_indexing_spans();
        }
        typename dynarray<T,Allocator>::rand_access_iterator pos(indexing_vec_[row_pos].data());
        data_vec_.insert(pos, container.begin(),container.end());
        for (size_type i =  size() - 1; i > row_pos; --i){
            indexing_vec_[i] = indexing_vec_[i-1];
        }
        indexing_vec_[row_pos].resize(new_elements_count);
        #pragma omp parallel for
        for (size_type i = row_pos+1; i < size(); ++i)
        {
            indexing_vec_[i].change_span_ptr(indexing_vec_[i].data() + new_elements_count);
        }
    }
    

    template<typename T,typename Allocator> 
    template<typename Container, typename index_container>
    inline auto r2darray<T,Allocator>::buffered_insert(const Container & insert_elements,const  index_container & row_indices,const index_container & column_indices)     
        -> type_<void,
            decltype(std::declval<Container>().begin()),
            decltype(std::declval<Container>().end()),
            decltype(std::declval<Container>().size()),
            decltype(std::declval<index_container>().begin()),
            decltype(std::declval<index_container>().end()),
            decltype(std::declval<index_container>().size()),
            decltype(static_cast<int>(*(std::declval<index_container>().begin())))>
    {
        const size_type new_elements_count = std::min({insert_elements.size(),row_indices.size(),column_indices.size()}); // minimum of the size of the 3 input containers
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<size_type> s_allocator_type;
        s_allocator_type s_alloc(cap_alloc_.y());
        size_type * insert_data_vec_idx = reinterpret_cast<size_type*>(std::allocator_traits<s_allocator_type>::allocate(s_alloc,new_elements_count));         // for storing equivalent index for the flat dynarray
        if (data_vec_.capacity() < data_vec_.size() + new_elements_count){
            data_vec_.grow_reserve(data_vec_.size() + new_elements_count);     // any pointer invalidation happens here
            const difference_type offset = data_vec_.data()-indexing_vec_[0].data();  
            indexing_vec_[0].change_span_ptr(data_vec_.data());
            #pragma omp parallel for
            for (size_type i = 1; i < size(); ++i){
                indexing_vec_[i].change_span_ptr(indexing_vec_[i].data()+offset);
            }
        }
        auto row_it = row_indices.begin();
        auto col_it = column_indices.begin();
        for (size_type i = 0; i < new_elements_count; ++i)
        {
            insert_data_vec_idx[i] = indexing_vec_[*row_it].data() - data_vec_.data() + *col_it;
            #pragma omp parallel for
            for (size_type j = *row_it + 1; j < size(); ++j)
            {
                indexing_vec_[j].change_span_ptr(indexing_vec_[j].data() + 1);
            }
            indexing_vec_[*row_it].resize(indexing_vec_[*row_it].size() + 1);
            ++row_it;
            ++col_it;
        }
        data_vec_.buffered_insert(insert_elements,insert_data_vec_idx,new_elements_count);
        std::allocator_traits<s_allocator_type>::deallocate(s_alloc,reinterpret_cast<typename s_allocator_type::pointer>(insert_data_vec_idx),new_elements_count);
    }

    template<typename T,typename Allocator> 
    template<typename index_container>
    inline auto r2darray<T,Allocator>::buffered_erase(const index_container & row_indices,const index_container & column_indices)            
        -> type_<void,
            decltype(std::declval<index_container>().begin()),
            decltype(std::declval<index_container>().end()),
            decltype(std::declval<index_container>().size()),
            decltype(static_cast<int>(*(std::declval<index_container>().begin())))>
    {
        const size_type erase_elements_count = (row_indices.size() > column_indices.size()) ? row_indices.size():column_indices.size();
        typedef typename std::allocator_traits<allocator_type>::rebind_alloc<size_type> s_allocator_type;
        s_allocator_type s_alloc(cap_alloc_.y());
        size_type * erase_data_vec_idx = reinterpret_cast<size_type*>(std::allocator_traits<s_allocator_type>::allocate(s_alloc,erase_elements_count));         // for storing equivalent index for the flat dynarray
        auto row_it = row_indices.begin();
        auto col_it = column_indices.begin();
        for (size_type i = 0; i < erase_elements_count; ++i)
        {
            erase_data_vec_idx[i] = indexing_vec_[*row_it].data() - data_vec_.data() + *col_it;
            #pragma omp parallel for
            for (size_type j = *row_it + 1; j < size(); ++j)
            {
                indexing_vec_[j].change_span_ptr(indexing_vec_[j].data() - 1);
            }
            indexing_vec_[*row_it].resize(indexing_vec_[*row_it].size() - 1);
            ++row_it;
            ++col_it;
        }
        data_vec_.buffered_erase(erase_data_vec_idx,erase_elements_count);
        std::allocator_traits<s_allocator_type>::deallocate(s_alloc,reinterpret_cast<typename s_allocator_type::pointer>(erase_data_vec_idx),erase_elements_count);    
    }
#endif
}
#endif