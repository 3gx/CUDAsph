#ifndef _DREAL_H_
#define _DREAL_H_

struct dreal {
  float v[2];
};

#ifdef _DEVICE_CODE_

__device__ inline dreal dadd(dreal x, dreal y) {
  float t1 = x.v[0] + y.v[0];
  float e  = t1 - x.v[0];
  float t2 = ((y.v[0] - e) + (x.v[0] - (t1 - e))) + x.v[1] + y.v[1];
  
  dreal c;
  c.v[0] = e = t1 + t2;
  c.v[1] = t2 - (e - t1);
  return c;
}

__device__ inline dreal dadd(dreal x, float y) {
  float t1 = x.v[0] + y;
  float e  = t1 - x.v[0];
  float t2 = ((y - e) + (x.v[0] - (t1 - e))) + x.v[1];
  
  dreal c;
  c.v[0] = e = t1 + t2;
  c.v[1] = t2 - (e - t1);
  return c;
}

__device__ inline dreal dsub(dreal x, dreal y) {
  float t1 = x.v[0] - y.v[0];
  float e  = t1 - x.v[0];
  float t2 = ((-y.v[0] - e) + (x.v[0] - (t1 - e))) + x.v[1] - y.v[1];
  
  dreal c;
  c.v[0] = e = t1 + t2;
  c.v[1] = t2 - (e - t1);
  return c;
}

__device__ inline dreal dsub(dreal x, float y) {
  float t1 = x.v[0] - y;
  float e  = t1 - x.v[0];
  float t2 = ((-y - e) + (x.v[0] - (t1 - e))) + x.v[1];
  
  dreal c;
  c.v[0] = e = t1 + t2;
  c.v[1] = t2 - (e - t1);
  return c;
}

__device__ inline dreal dmul(dreal x, dreal y) {
  // This splits dsa(1) and dsb(1) into high-order and low-order words.
  float cona = x.v[0] * 8193.0f;
  float conb = y.v[0] * 8193.0f;
  float sa1 = cona - (cona - x.v[0]);
  float sb1 = conb - (conb - y.v[0]);
  float sa2 = x.v[0] - sa1;
  float sb2 = y.v[0] - sb1;

	// Multilply x.v[0] * y.v[0] using Dekker's method.
	float c11 = x.v[0] * y.v[0];
	float c21 = (((sa1 * sb1 - c11) + sa1 * sb2) + sa2 * sb1) + sa2 * sb2;

    // Compute x.v[0] * y.v[1]+x.v[1]* y.v[0] (only high-order word is needed).
    float c2 = x.v[0] * y.v[1]+x.v[1]* y.v[0];

    // Compute (c11, c21) + c2 using Knuth's trick, also adding low-order product.
    float t1 = c11 + c2;
    float e = t1 - c11;
    float t2 = ((c2 - e) + (c11 - (t1 - e))) + c21 +x.v[1]* y.v[1];

    // The result is t1 + t2, after normalization.
    dreal c;
    c.v[0] = e = t1 + t2;
    c.v[1] = t2 - (e - t1);
    return c;
} // dsmul

#else

inline dreal convert_to_dreal(double y) {
  dreal x;
  x.v[0] = (float)y;
  x.v[1] = (float)(y - x.v[0]);
  return x;
}

inline dreal dadd(dreal x, dreal y) {
  float t1 = x.v[0] + y.v[0];
  float e  = t1 - x.v[0];
  float t2 = ((y.v[0] - e) + (x.v[0] - (t1 - e))) + x.v[1] + y.v[1];
  
  dreal c;
  c.v[0] = e = t1 + t2;
  c.v[1] = t2 - (e - t1);
  return c;
}

inline dreal dadd(dreal x, float y) {
  float t1 = x.v[0] + y;
  float e  = t1 - x.v[0];
  float t2 = ((y - e) + (x.v[0] - (t1 - e))) + x.v[1];
  
  dreal c;
  c.v[0] = e = t1 + t2;
  c.v[1] = t2 - (e - t1);
  return c;
}

inline dreal dsub(dreal x, dreal y) {
  float t1 = x.v[0] - y.v[0];
  float e  = t1 - x.v[0];
  float t2 = ((-y.v[0] - e) + (x.v[0] - (t1 - e))) + x.v[1] - y.v[1];
  
  dreal c;
  c.v[0] = e = t1 + t2;
  c.v[1] = t2 - (e - t1);
  return c;
}

#endif


#endif
