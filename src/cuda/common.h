#ifndef __PAGMO_CUDA_COMMON_H_
#define __PAGMO_CUDA_COMMON_H_

// common kernel and device functions/classes

template <typename fty>
struct nop_functor;

template <typename fty>
struct scale_functor;


#define ADJUSTED_TX(TPB, BX, TX) ((BX) * (TPB) + (TX))
#define RAW_TX(BDX, BX, TX) ((BX) * (BDX) + (TX))

#define BLOCK_ADJUSTED_TX(TPB) ADJUSTED_TX(TPB,0,threadIdx.x)
#define GLOBAL_ADJUSTED_TX(TPB) ADJUSTED_TX(TPB,blockIdx.x,threadIdx.x)

#define BLOCK_RAW_TX() RAW_TX(blockDim.x, 0, threadIdx.x)
#define GLOBAL_RAW_TX() RAW_TX(blockDim.x, blockIdx.x, threadIdx.x)



#define BLOCK_TASKS(TPB) (TPB)
#define BLOCK_POINTS(TPB, TS) (BLOCK_TASKS(TPB) / (TS))
#define BLOCK_INDIVIDUALS(TPB, TS, PTS) (BLOCK_TASKS(TPB) / ((TS)*(PTS)))


#define GLOBAL_INDIV_ID(TPB, TS, PTS) (GLOBAL_ADJUSTED_TX(TPB) / (PTS * TS))
#define BLOCK_INDIV_ID(TPB,TS,PTS) (BLOCK_ADJUSTED_TX(TPB) / (PTS * TS))

#define GLOBAL_POINT_ID(TPB, TS) (GLOBAL_ADJUSTED_TX(TPB) / (TS))
#define BLOCK_POINT_ID(TPB,TS) (BLOCK_ADJUSTED_TX(TPB) / (TS))

#define GLOBAL_TASK_ID(TPB) GLOBAL_ADJUSTED_TX(TPB)
#define BLOCK_TASK_ID(TPB) BLOCK_ADJUSTED_TX(TPB)


#define IS_VALID_FOR_BLOCK(TPB,TS,PTS) (BLOCK_INDIV_ID(TPB, TS, PTS) < BLOCK_INDIVIDUALS(TPB,TS,PTS) && BLOCK_TASK_ID(TPB) < BLOCK_TASKS(TPB))

#endif
