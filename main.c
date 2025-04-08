#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "parameter.h"
#include <immintrin.h>
#include <omp.h>
// #include </home/wells/OpenBLAS/OpenBLAS/cblas.h>
#define int size_t
//define macro so the matrixes are stored in column-major, which is used by Fortan and OpenBLAS
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]
#define min(a, b) (((a) < (b)) ? (a) : (b))

void printArray(int m,int n,float *a,int lda);
void matmul_plain(int m,int n,int k,float *a,int lda,float *b,int ldb,float *c,int ldc);
void matmul_plain_SIMD(int m,int n,int k,float *a,int lda,float *b,int ldb,float *c,int ldc);
void AddDot(int k,float *a,int lda,float *b,int ldb,float *gamma);
void matmul_tech(int m,int n,int k,float *a,int lda,float *b,int ldb,float *c,int ldc);
void AddMM(int k,float *a,int lda,float *b,int ldb,float *c,int ldc);
void generate_1D_Matrix(int m,int n,float *a,int lda);
inline void packA(int m,int k,float *x,int ldx,int offset,float *pack);
inline void packB(int n,int k,float *y,int ldy,int offset,float *pack);
float *malloc_aligned(int m,int n,int size);
void matmul_gemm(int m,int n,int k,float *a,int lda,float *b,int ldb,float *c,int ldc);
void AddDot4x4(int k,float *a,int lda,float *b,int ldb,float *c,int ldc);
void matmul_gemm_mk(int m,int n,int k,float *a,int lda,float *b,int ldb,float *c,int ldc);
void matmul_gemm_omp(int m,int n,int k,float *a,int lda,float *b,int ldb,float *c,int ldc);
inline void thread_shared(int n,int bf,int *start,int *end);
double dclock();
unsigned main()
{
    
    int p,m, n, k,lda, ldb, ldc,rep;//m,n,k is the real size of the matrix while lda,ldb,ldc are the "leading dimension"
    double dtime,gflops;
    float *a,*b, *c;
    // printf("MY_MMult = [\n");
    for (p = START; p <= END; p+=INC)
    {
        m=n=k=p;
        gflops = 2.0 * m * n * k * 1.0e-09;
        lda = (LDA == -1 ? m : LDA);
        ldb = (LDB == -1 ? k : LDB);
        ldc = (LDC == -1 ? m : LDC);
        /* Allocate space for the matrices */
        /* Note: I create an extra column in A to make sure that
        prefetching beyond the matrix does not cause a segfault */
        a = (float*)malloc(lda*(k+1)*sizeof(float));
        b = (float*)malloc(ldb*n*sizeof(float));
        c = (float*)malloc(ldc*n*sizeof(float));
        generate_1D_Matrix(m,k,a,lda);
        generate_1D_Matrix(k,n,b,ldb);
        generate_1D_Matrix(m,n,c,ldc);
        // printArray(m,k,a,lda);
        dtime = dclock();
        /*Run the optimized implementation of matrix multiplication*/
        // matmul_plain(m,n,k,a,lda,b,ldb,c,ldc);
        // cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,a,lda,b,ldb,0,c,ldc);
        // matmul_plain_SIMD(m,n,k,a,lda,b,ldb,c,ldc);
        // matmul_tech(m,n,k,a,lda,b,ldb,c,ldc);
        // matmul_gemm(m,n,k,a,lda,b,ldb,c,ldc);
        // matmul_gemm_mk(m,n,k,a,lda,b,ldb,c,ldc);
        matmul_gemm_omp(m,n,k,a,lda,b,ldb,c,ldc);
        dtime = dclock()-dtime;
        printf("%zu %zu %le \n",p,p,gflops/dtime);
        // printArray(m,n,c,ldc);
        fflush(stdout);
        free(a);free(b);free(c);
    }
    // printf("];\n");
}
void printArray(int m,int n,float *a,int lda){
    if(a==NULL) return;
    int i,j;
    for(j=0;j<n;j++){
        for(i=0;i<m;i++){
            printf("%le \n",a[j*lda+i]);
        }
        printf("\n");
    }
}
void generate_1D_Matrix(int m,int n,float *a,int lda){
    if(a==NULL) return;
    srand(time(NULL));
    int i,j;
    for(j=0;j<n;j++){
        for(i=0;i<m;i++){
            A(i,j) = (float)rand()/RAND_MAX*1000;
        }
    }
}
/*The plain implementation of C+=A*B*/
void matmul_plain(int m,int n,int k,float *a,int lda,float *b,int ldb,float *c,int ldc){
    if(a==NULL||b==NULL||c==NULL) return;
    int i,j,p;
    for(i=0;i<n;i++){
        for(j=0;j<k;j++){
            for(p=0;p<m;p++){
                C(p,i) = C(p,i) + A(p,j)*B(j,i);
            }
        }
    }
}
void matmul_plain_SIMD(int m,int n,int k,float *a,int lda,float *b,int ldb,float *c,int ldc){
    if(a==NULL||b==NULL||c==NULL) return;
    if(m%8!=0||n%8!=0||k%8!=0) fprintf(stderr,"Input size must be multiplier of 8!");
    int i,j,p;
    __m256 x,y;
    __m256 gamma=_mm256_setzero_ps();
    for(i=0;i<n;i++){
        for(j=0;j<k;j++){
            y=_mm256_set1_ps(B(j,i)); //store B[i,j] to all element of y(the jth column)
            for(p=0;p<m;p+=8){
                x=_mm256_loadu_ps(&A(p,j));
                gamma=_mm256_add_ps(gamma,_mm256_mul_ps(x,y));
                _mm256_storeu_ps(&C(p,i),gamma);
            }
        }
    }
}
void matmul_tech(int m,int n,int k,float *a,int lda,float *b,int ldb,float *c,int ldc){
    if(a==NULL||b==NULL||c==NULL) return;
    int i,j;
    for(j=0;j<n;j+=DGEMM_NRr){
        for(i=0;i<m;i+=DGEMM_MRr){
            AddMM(k,&A(i,0),lda,&B(0,j),ldb,&C(i,j),ldc);
        }
    }
    
}
void AddDot(int k,float *a,int lda,float *b,int ldb,float *gamma){
    if(a==NULL||b==NULL||gamma==NULL) return;
    int p;
    for(p=0;p<k;p++){
        *gamma += A(0,p)*B(p,0);//A's row multiply by B's column
    }
}
void AddMM(int k,float *a,int lda,float *b,int ldb,float *c,int ldc){
    if(a==NULL||b==NULL||c==NULL) return;
    int h,l;
    for(h=0;h<DGEMM_NRr;h++){
        for(l=0;l<DGEMM_MRr;l++){
            AddDot(k,&A(l,0),lda,&B(0,h),ldb,&C(l,h));
        }
    }
}
inline void packA(int m,int k,float *x,int ldx,int offset,float *pack){//data should be accessed by column first
    if(x==NULL||pack==NULL) return;
    int i,p;
    float *a_ptr[DGEMM_MRr];
    for(i=0;i<m;i++){a_ptr[i]=x+(offset+i);}
    for(i=m;i<DGEMM_MRr;i++){a_ptr[i]=x+(offset+0);}
    for(p=0;p<k;p++){
        for(i=0;i<DGEMM_MRr;i++){
            *pack = *a_ptr[i];
            pack ++;
            a_ptr[i] = a_ptr[i]+ldx;
        }
    }
}
inline void packB(int n,int k,float *y,int ldy,int offset,float *pack){//data should be accessed by row first
    if(y==NULL||pack==NULL) return;
    int j,p;
    float *b_ptr[DGEMM_NRr];
    for(j=0;j<n;j++){b_ptr[j]=y+ldy*(offset+j);}
    for(j=n;j<DGEMM_NRr;j++){b_ptr[j]=y+ldy*(offset+0);}
    for(p=0;p<k;p++){
        for(j=0;j<DGEMM_NRr;j++){
            *pack++ = *b_ptr[j]++;
        }
    }
}

float *malloc_aligned(int m,int n,int size){
    float *ptr;
    int e;
    e = posix_memalign((void**)&ptr,(int)32,size*m*n);//needs a pointer to the pointer to reallocate the memory
    if(e){
        printf("aligned memory aligned fails");
        exit(1);
    }
    return ptr;
}

void matmul_gemm(int m,int n,int k,float *a,int lda,float *b,int ldb,float *c,int ldc){
    if(a==NULL||b==NULL||c==NULL) return;
    int i,j,p;
    int ic,ib,jc,jb,pc,pb,jrb,irb;//control for outer three loop
    int ir,jr,kr;
    float *_A_, *_B_;
    char *str;
    //Allocate space for packs, align the space.
    _A_ = malloc_aligned((DGEMM_MC+1),DGEMM_KC,sizeof(float));//for prefetching, add one row
    _B_ = malloc_aligned(DGEMM_KC,(DGEMM_NC+1),sizeof(float));//for prefetching, add one column
    for(jc=0;jc<n;jc+=DGEMM_NC){//Loop5
        jb=min(n-jc,DGEMM_NC);//ensure that the remaining doesn't exceed the columns of B when packing
        for(pc=0;pc<k;pc+=DGEMM_KC){//Loop4
            pb=min(k-pc,DGEMM_KC);
            for(j=0;j<jb;j+=DGEMM_NRr){//PackB,iterate through column of B
                packB(min(jb-j,DGEMM_NRr),pb,&b[pc],ldb,jc+j,&_B_[j*pb]);
            }
            for(ic=0;ic<m;ic+=DGEMM_MC){//Loop3
                ib=min(m-ic,DGEMM_MC);
                for(i=0;i<ib;i+=DGEMM_MRr){//PackA,iterate through row of A
                    packA(min(ib-i,DGEMM_MRr),pb,&a[pc*lda],m,ic+i,&_A_[i*pb]);
                }
                for(jr=0;jr<jb;jr+=DGEMM_NRr){//Loop2
                    jrb=min(jb-jr,DGEMM_NRr);
                    for(ir=0;ir<ib;ir+=DGEMM_MRr){//Loop1
                        irb=min(ib-ir,DGEMM_MRr);
                        for(kr=0;kr<pb;kr++){//Loop0--micro-kernal
                            C(ir,jr) = _A_[ir*kr]*_B_[jr*kr] + C(ir,jr);
                        }
                    }
                }
            }
        }
    }
    free(_A_);
    free(_B_);
}

void matmul_gemm_mk(int m,int n,int k,float *a,int lda,float *b,int ldb,float *c,int ldc){
    if(a==NULL||b==NULL||c==NULL) return;
    int i,j,p;
    int ic,ib,jc,jb,pc,pb,jrb,irb;//control for outer three loop
    int ir,jr,kr;
    float *_A_, *_B_;
    char *str;
    //Allocate space for packs, align the space.
    _A_ = malloc_aligned((DGEMM_MC+1),DGEMM_KC,sizeof(float));//for prefetching, add one row
    _B_ = malloc_aligned(DGEMM_KC,(DGEMM_NC+1),sizeof(float));//for prefetching, add one column
    for(jc=0;jc<n;jc+=DGEMM_NC){//Loop5
        jb=min(n-jc,DGEMM_NC);//ensure that the remaining doesn't exceed the columns of B when packing
        for(pc=0;pc<k;pc+=DGEMM_KC){//Loop4
            pb=min(k-pc,DGEMM_KC);
            for(j=0;j<jb;j+=DGEMM_NRr){//PackB,iterate through column of B
                packB(min(jb-j,DGEMM_NRr),pb,&b[pc],ldb,jc+j,&_B_[j*pb]);
            }
            for(ic=0;ic<m;ic+=DGEMM_MC){//Loop3
                ib=min(m-ic,DGEMM_MC);
                for(i=0;i<ib;i+=DGEMM_MRr){//PackA,iterate through row of A
                    packA(min(ib-i,DGEMM_MRr),pb,&a[pc*lda],m,ic+i,&_A_[i*pb]);
                }
                //marco-kernal
                for(jr=0;jr<jb;jr+=DGEMM_NRr){//Loop2
                    jrb=min(jb-jr,DGEMM_NRr);
                    for(ir=0;ir<ib;ir+=DGEMM_MRr){//Loop1
                        irb=min(ib-ir,DGEMM_MRr);
                        AddDot4x4(pb,&_A_[ir*pb],lda,&_B_[jr*pb],ldb,&C(ir,jr),ldc);
                    }
                }

            }
        }
    }
    free(_A_);
    free(_B_);
}

void AddDot4x4(int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {//need to set DGEMM_MRr to 4
    /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 

     And now we use vector registers and instructions */
    __m256 c_00_c_10_vreg = _mm256_setzero_ps();
    __m256 c_01_c_11_vreg = _mm256_setzero_ps();
    __m256 c_02_c_12_vreg = _mm256_setzero_ps();
    __m256 c_03_c_13_vreg = _mm256_setzero_ps();
    __m256 c_20_c_30_vreg = _mm256_setzero_ps();
    __m256 c_21_c_31_vreg = _mm256_setzero_ps();
    __m256 c_22_c_32_vreg = _mm256_setzero_ps();
    __m256 c_23_c_33_vreg = _mm256_setzero_ps();

    for (int p = 0; p < k; p++) {
        __m256 a_0p_a_1p_vreg = _mm256_load_ps(a);
        __m256 a_2p_a_3p_vreg = _mm256_load_ps(a + 2);
        a += 4;

        __m256 b_p0_vreg = _mm256_broadcast_ss(b);
        __m256 b_p1_vreg = _mm256_broadcast_ss(b + 1);
        __m256 b_p2_vreg = _mm256_broadcast_ss(b + 2);
        __m256 b_p3_vreg = _mm256_broadcast_ss(b + 3);
        b += 4;
        c_00_c_10_vreg = _mm256_add_ps(c_00_c_10_vreg, _mm256_mul_ps(a_0p_a_1p_vreg, b_p0_vreg));
        c_01_c_11_vreg = _mm256_add_ps(c_01_c_11_vreg, _mm256_mul_ps(a_0p_a_1p_vreg, b_p1_vreg));
        c_02_c_12_vreg = _mm256_add_ps(c_02_c_12_vreg, _mm256_mul_ps(a_0p_a_1p_vreg, b_p2_vreg));
        c_03_c_13_vreg = _mm256_add_ps(c_03_c_13_vreg, _mm256_mul_ps(a_0p_a_1p_vreg, b_p3_vreg));

        c_20_c_30_vreg = _mm256_add_ps(c_20_c_30_vreg, _mm256_mul_ps(a_2p_a_3p_vreg, b_p0_vreg));
        c_21_c_31_vreg = _mm256_add_ps(c_21_c_31_vreg, _mm256_mul_ps(a_2p_a_3p_vreg, b_p1_vreg));
        c_22_c_32_vreg = _mm256_add_ps(c_22_c_32_vreg, _mm256_mul_ps(a_2p_a_3p_vreg, b_p2_vreg));
        c_23_c_33_vreg = _mm256_add_ps(c_23_c_33_vreg, _mm256_mul_ps(a_2p_a_3p_vreg, b_p3_vreg));

    }

    _mm256_storeu_ps(&c[0 * ldc], _mm256_loadu_ps(&c[0 * ldc]) + c_00_c_10_vreg);
    _mm256_storeu_ps(&c[1 * ldc], _mm256_loadu_ps(&c[1 * ldc]) + c_01_c_11_vreg);
    _mm256_storeu_ps(&c[2 * ldc], _mm256_loadu_ps(&c[2 * ldc]) + c_02_c_12_vreg);
    _mm256_storeu_ps(&c[3 * ldc], _mm256_loadu_ps(&c[3 * ldc]) + c_03_c_13_vreg);
    _mm256_storeu_ps(&c[4 * ldc], _mm256_loadu_ps(&c[4 * ldc]) + c_20_c_30_vreg);
    _mm256_storeu_ps(&c[5 * ldc], _mm256_loadu_ps(&c[5 * ldc]) + c_21_c_31_vreg);
    _mm256_storeu_ps(&c[6 * ldc], _mm256_loadu_ps(&c[6 * ldc]) + c_22_c_32_vreg);
    _mm256_storeu_ps(&c[7 * ldc], _mm256_loadu_ps(&c[7 * ldc]) + c_23_c_33_vreg);
}

void matmul_gemm_omp(int m,int n,int k,float *a,int lda,float *b,int ldb,float *c,int ldc){
    if(a==NULL||b==NULL||c==NULL) return;
    int i,j,p;
    int ic,ib,jc,jb,pc,pb,jrb,irb;//control for outer three loop
    int ir,jr,kr;
    float *_A_, *_B_;
    char *str;
    //Allocate space for packs, align the space.
    _A_ = malloc_aligned((DGEMM_MC+1)*INC,DGEMM_KC,sizeof(float));//for prefetching, add one row
    _B_ = malloc_aligned(DGEMM_KC,(DGEMM_NC+1),sizeof(float));//for prefetching, add one column
    for(jc=0;jc<n;jc+=DGEMM_NC){//Loop5
        jb=min(n-jc,DGEMM_NC);//ensure that the remaining doesn't exceed the columns of B when packing
        for(pc=0;pc<k;pc+=DGEMM_KC){//Loop4
            pb=min(k-pc,DGEMM_KC);
            // #pragma omp parallel num_threads(INCC) private(jr)
            for(j=0;j<jb;j+=DGEMM_NRr){//PackB,iterate through column of B
                packB(min(jb-j,DGEMM_NRr),pb,&b[pc],ldb,jc+j,&_B_[j*pb]);
            }
            #pragma omp parallel num_threads(INCC) private(jr,jrb,ir,irb,kr)//multithread for Loop3
                {
                int tid = omp_get_thread_num(); 
                int my_start,my_end;
                thread_shared(m,DGEMM_MRr,&my_start,&my_end);//compute blocks of A by increasement mr and multithreading
                for(ic=my_start;ic<my_end;ic+=DGEMM_MC){//Loop3
                ib=min(my_end-ic,DGEMM_MC);
                for(i=0;i<ib;i+=DGEMM_MRr){//PackA,iterate through row of A
                    packA(min(ib-i,DGEMM_MRr),pb,&a[pc*lda],m,ic+i,&_A_[i*pb+tid*DGEMM_MC*pb]);//computer how many A has been packed by multithread
                }
                    for(jr=0;jr<jb;jr+=DGEMM_NRr){//Loop2
                    jrb=min(jb-jr,DGEMM_NRr);
                    for(ir=0;ir<ib;ir+=DGEMM_MRr){//Loop1
                        irb=min(ib-ir,DGEMM_MRr);
                        for(kr=0;kr<pb;kr++){//Loop0--micro-kernal
                            C(ir,jr) = _A_[ir*kr + tid*DGEMM_MC*pb]*_B_[jr*kr] + C(ir,jr);//computer how many A has been packed by multithread
                        }
                    }
                
                }              
            }
                }
        }
    }
    free(_A_);
    free(_B_);
}

inline void thread_shared(int n,int bf,int *start,int *end){
    if(start==NULL||end==NULL) return;
    int way =omp_get_num_threads();
    int id = omp_get_thread_num();
    //thread partitioning
    //we partition the space between all_start and
	// all_end into n_way partitions, each a multiple of block_factor
	// with the exception of the one partition that recieves the
	// "edge" case (if applicable).
    int all_start=0;
    int all_end=n;
    int size=all_end-all_start;
    int whole=size/bf;
    int left=size%bf;
    int lo=whole/way;int hi=whole/way;
    //differentiate between edge cases
    int b_th_lo=whole%way;
    if(lo!=0) lo++;
    int size_lo=lo*bf,size_hi=hi*bf;
    // Precompute the starting indices of the low and high groups.
    int lo_start=all_start,hi_start=all_start+b_th_lo*size_lo;
    // Compute the start and end of individual threads' ranges
	// as a function of their work_ids and also the group to which
	// they belong (low or high).
    if(id<b_th_lo){
        *start=lo_start+id*size_lo;
        *end=lo_start+(id+1)*size_lo;
    }
    else{*start=hi_start+(id-b_th_lo)*size_hi;
    *end=hi_start+(id-b_th_lo+1)*size_hi;
    if(id==way-1) *end+=left;}
}












