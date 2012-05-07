#include "CUDAsph.h"

#define BITS_PER_DIMENSION 18
typedef long long peanokey;

struct peano_struct {
  peanokey key;
  int  value;
};

static int quadrants[24][2][2][2] = {
  /* rotx=0, roty=0-3 */
  {{{0, 7}, {1, 6}}, {{3, 4}, {2, 5}}},
  {{{7, 4}, {6, 5}}, {{0, 3}, {1, 2}}},
  {{{4, 3}, {5, 2}}, {{7, 0}, {6, 1}}},
  {{{3, 0}, {2, 1}}, {{4, 7}, {5, 6}}},
  /* rotx=1, roty=0-3 */
  {{{1, 0}, {6, 7}}, {{2, 3}, {5, 4}}},
  {{{0, 3}, {7, 4}}, {{1, 2}, {6, 5}}},
  {{{3, 2}, {4, 5}}, {{0, 1}, {7, 6}}},
  {{{2, 1}, {5, 6}}, {{3, 0}, {4, 7}}},
  /* rotx=2, roty=0-3 */
  {{{6, 1}, {7, 0}}, {{5, 2}, {4, 3}}},
  {{{1, 2}, {0, 3}}, {{6, 5}, {7, 4}}},
  {{{2, 5}, {3, 4}}, {{1, 6}, {0, 7}}},
  {{{5, 6}, {4, 7}}, {{2, 1}, {3, 0}}},
  /* rotx=3, roty=0-3 */
  {{{7, 6}, {0, 1}}, {{4, 5}, {3, 2}}},
  {{{6, 5}, {1, 2}}, {{7, 4}, {0, 3}}},
  {{{5, 4}, {2, 3}}, {{6, 7}, {1, 0}}},
  {{{4, 7}, {3, 0}}, {{5, 6}, {2, 1}}},
  /* rotx=4, roty=0-3 */
  {{{6, 7}, {5, 4}}, {{1, 0}, {2, 3}}},
  {{{7, 0}, {4, 3}}, {{6, 1}, {5, 2}}},
  {{{0, 1}, {3, 2}}, {{7, 6}, {4, 5}}},
  {{{1, 6}, {2, 5}}, {{0, 7}, {3, 4}}},
  /* rotx=5, roty=0-3 */
  {{{2, 3}, {1, 0}}, {{5, 4}, {6, 7}}},
  {{{3, 4}, {0, 7}}, {{2, 5}, {1, 6}}},
  {{{4, 5}, {7, 6}}, {{3, 2}, {0, 1}}},
  {{{5, 2}, {6, 1}}, {{4, 3}, {7, 0}}}
};

static int rotxmap_table[24] = { 4, 5, 6, 7, 8, 9, 10, 11,
  12, 13, 14, 15, 0, 1, 2, 3, 17, 18, 19, 16, 23, 20, 21, 22
};

static int rotymap_table[24] = { 1, 2, 3, 0, 16, 17, 18, 19,
  11, 8, 9, 10, 22, 23, 20, 21, 14, 15, 12, 13, 4, 5, 6, 7
};

static int rotx_table[8]  = { 3, 0, 0, 2, 2, 0, 0, 1 };
static int roty_table[8]  = { 0, 1, 1, 2, 2, 3, 3, 0 };

static int sense_table[8] = { -1, -1, -1, +1, +1, -1, -1, -1 };

// static int flag_quadrants_inverse = 1;
// static char quadrants_inverse_x[24][8];
// static char quadrants_inverse_y[24][8];
// static char quadrants_inverse_z[24][8];

/*! This function computes a Peano-Hilbert key for an integer triplet (x,y,z),
 *  with x,y,z in the range between 0 and 2^bits-1.
 */

peanokey peano_hilbert_key(int x, int y, int z, int bits) {
  int i, quad, bitx, bity, bitz;
  int mask, rotation, rotx, roty, sense;
  peanokey key;
  
  mask = 1 << (bits - 1);
  key = 0;
  rotation = 0;
  sense = 1;

  for(i = 0; i < bits; i++, mask >>= 1)
    {
      bitx = (x & mask) ? 1 : 0;
      bity = (y & mask) ? 1 : 0;
      bitz = (z & mask) ? 1 : 0;
      
      quad = quadrants[rotation][bitx][bity][bitz];

      key <<= 3;
      key += (sense == 1) ? (quad) : (7 - quad);

      rotx = rotx_table[quad];
      roty = roty_table[quad];
      sense *= sense_table[quad];

      while(rotx > 0)
	{
	  rotation = rotxmap_table[rotation];
	  rotx--;
	}

      while(roty > 0)
	{
	  rotation = rotymap_table[rotation];
	  roty--;
	}
    }

  return key;
}


int compare_peanokey(const void *a, const void *b) {
  if(((struct peano_struct *) a)->key < (((struct peano_struct *) b)->key))
    return -1;
  
  if(((struct peano_struct *) a)->key > (((struct peano_struct *) b)->key))
    return +1;
  
  return 0;
}


void CUDAsph::sort_bodies() {
  
  float4 r_min = hst.bodies_pos[0];
  float4 r_max  = r_min;
  for (int i = 0; i < n_bodies; i++) {
    float4 pos = hst.bodies_pos[i];
    r_min.x = min(r_min.x, pos.x);
    r_min.y = min(r_min.y, pos.y);
    r_min.z = min(r_min.z, pos.z);

    r_max.x = max(r_max.x, pos.x);
    r_max.y = max(r_max.y, pos.y);
    r_max.z = max(r_max.z, pos.z);
  }
  float size = max(r_max.z - r_min.z, max(r_max.y - r_min.y, r_max.x - r_min.x));

  float domain_fac = 1.0 / size * (((peanokey)1) << (BITS_PER_DIMENSION));

  peano_struct *keys = (peano_struct*)malloc(n_bodies * sizeof(peano_struct));

  float4 *bodies_pos_0  = (float4*)malloc(n_bodies * sizeof(float4));
  float4 *bodies_vel_0  = (float4*)malloc(n_bodies * sizeof(float4));
  float4 *bodies_dots_0 = (float4*)malloc(n_bodies * sizeof(float4));
  float4 *hydro_data_0  = (float4*)malloc(n_bodies * sizeof(float4));
  float4 *dots_data_0   = (float4*)malloc(n_bodies * sizeof(float4));
  float4 *hydro_data2_0 = (float4*)malloc(n_bodies * sizeof(float4));
  float4 *grav_data_0   = (float4*)malloc(n_bodies * sizeof(float4));

  for (int i = 0; i < n_bodies; i++) {
    bodies_pos_0[i]  = hst.bodies_pos[i];
    bodies_vel_0[i]  = hst.bodies_vel[i];
    bodies_dots_0[i] = hst.bodies_dots[i];
    hydro_data_0[i]  = hst.hydro_data[i];
    dots_data_0[i]   = hst.dots_data[i];
    hydro_data2_0[i] = hst.hydro_data2[i];
    grav_data_0[i]   = hst.grav_data[i];
    
    keys[i].value = i;
    
    float4 pos = hst.bodies_pos[i];
    int x = (int)((pos.x - r_min.x) * domain_fac);
    int y = (int)((pos.y - r_min.y) * domain_fac);
    int z = (int)((pos.z - r_min.z) * domain_fac);
    keys[i].key   = peano_hilbert_key(x, y, z, BITS_PER_DIMENSION);
  }
  
  qsort(keys, n_bodies, sizeof(peano_struct), compare_peanokey);
  
  for (int i = 0; i < n_bodies; i++) {
    int j = keys[i].value;
    hst.bodies_pos[i]  = bodies_pos_0[j];
    hst.bodies_vel[i]  = bodies_vel_0[j];
    hst.bodies_dots[i] = bodies_dots_0[j];
    hst.hydro_data[i]  = hydro_data_0[j];
    hst.dots_data[i]   = dots_data_0[j];
    hst.hydro_data2[i] = hydro_data2_0[j];
    hst.grav_data[i]   = grav_data_0[j];
  }
  
  free(keys);  
  free(bodies_pos_0);
  free(bodies_vel_0);
  free(bodies_dots_0);
  free(hydro_data_0);
  free(dots_data_0);
  free(hydro_data2_0);
  free(grav_data_0);
}
