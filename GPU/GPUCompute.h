//  GPU/GPUCompute.h

#include "GPUEngine.h" 


#define SEARCH_MODE_ADDRESS 0
#define SEARCH_MODE_PUBKEY  1

__device__ PubKeyData* d_targets;
__device__ int d_numTargets;

__device__ __constant__ uint32_t d_searchInfo; 

__device__ __noinline__ void CheckPointPubKey(uint64_t *px, uint8_t py_is_odd, int32_t incr, uint32_t *out) {
    uint32_t target_y_is_odd = (py_is_odd << 1);

    for (int i = 0; i < d_numTargets; i++) {
        if (px[0] == d_targets[i].x[0] && px[1] == d_targets[i].x[1] &&
            px[2] == d_targets[i].x[2] && px[3] == d_targets[i].x[3]) {
            
            if (target_y_is_odd == (d_targets[i].parity << 1)) {
                uint32_t pos = atomicAdd(out, 1);
                
                uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
                uint32_t* item_ptr = out + (pos * ITEM_SIZE32 + 1);
                
                item_ptr[0] = tid;
                item_ptr[1] = (uint32_t)(incr << 16) | (uint32_t)(1 << 15);
                
                uint32_t* foundIndex_ptr = (uint32_t*)(&item_ptr[2]);
                *foundIndex_ptr = i;
            }
        }
    }
}

__device__ __noinline__ void CheckPoint(uint32_t *_h, int32_t incr, address_t *address, uint32_t *lookup32, uint32_t *out) {

  uint32_t   off;
  addressl_t  l32;
  address_t   pr0;
  address_t   hit;
  uint32_t   st;
  uint32_t   ed;
  uint32_t   mi;
  uint32_t   lmi;
  
    pr0 = *(address_t *)(_h);
    hit = address[pr0];

    if (hit) {
        if (lookup32) {
            off = lookup32[pr0];
            l32 = _h[0];
            st = off;
            ed = off + hit - 1;
            while (st <= ed) {
                mi = (st + ed) / 2;
                lmi = lookup32[mi];
                if (l32 < lmi) {
                    ed = mi - 1;
                }
                else if (l32 == lmi) {
                    goto addItem;
                }
                else {
                    st = mi + 1;
                }
            }
            return;
        }

    addItem:
        uint32_t pos = atomicAdd(out, 1);
        uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        uint32_t* item_ptr = out + (pos * ITEM_SIZE32 + 1);
        
        item_ptr[0] = tid;
        item_ptr[1] = (uint32_t)(incr << 16) | (uint32_t)(1 << 15);
        item_ptr[2] = _h[0];
        item_ptr[3] = _h[1];
        item_ptr[4] = _h[2];
        item_ptr[5] = _h[3];
        item_ptr[6] = _h[4];
    }
}
