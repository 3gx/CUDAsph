#ifndef __DEV_ART_VISC_CU_
#define __DEV_ART_VISC_CU_

#ifndef POW2
#define POW2(x) (x*x)
#endif

__device__ float2 dev_art_visc(float range_i, float range_j,
			       float ds, float drdv,
			       float4 hd_i,
			       float4 dd_i,
			       float4 hd_j,
			       float4 dd_j) {
  
  float c_i = sqrtf(5.0/3.0 * hd_i.y/hd_i.x);
  float c_j = sqrtf(5.0/3.0 * hd_j.y/hd_j.x);
  
  float art_visc = 0;
  float v_sig = (c_i + c_j);
  
  if (drdv < 0) {
    v_sig -= av_beta*drdv/ds;
    art_visc = drdv/((c_i + c_j)*ds);
    c_i = fabs(dd_i.x)/(fabs(dd_i.x) + dd_i.y + 0.0001 * c_i/(0.5*range_i));
    c_j = fabs(dd_j.x)/(fabs(dd_j.x) + dd_j.y + 0.0001 * c_j/(0.5*range_j));
    
    art_visc *= (c_i + c_j);
    art_visc = (hd_i.y/POW2(hd_i.x) + hd_j.y/POW2(hd_j.x)) *
      (-av_alpha * art_visc + av_beta * POW2(art_visc));
  }
  return make_float2(art_visc, v_sig);
}


#endif
