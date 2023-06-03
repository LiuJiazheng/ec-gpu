/*
 * Vector add
 */

#ifdef __NVCC__
  #define GET_GROUP_SIZE() gridDim.x
#else
  #define GET_GROUP_SIZE() get_work_dim()
#endif

KERNEL void {field}_vector_add(GLOBAL {field} *vec1,
                               GLOBAL {field} *vec2,
                               GLOBAL {field} *res,
                               int n
                               ) {{

    uint start_id = GET_GLOBAL_ID();
    uint period = GET_GROUP_SIZE() * GET_LOCAL_SIZE();

    for (uint i = start_id; i < n; i += period) {{
        res[i] = {field}_add(vec1[i], vec2[i]);
    }}
}} 

KERNEL void {field}_pointwise_add(GLOBAL {field} elem,
                                  GLOBAL {field} *vec,
                                  int n
                                  ) {{

    uint start_id = GET_GLOBAL_ID();
    uint period = GET_GROUP_SIZE() * GET_LOCAL_SIZE();

    for (uint i = start_id; i < n; i += period) {{
        vec[i] = {field}_add(elem, vec[i]);
    }}
}} 