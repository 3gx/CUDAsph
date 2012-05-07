#ifndef _KD_TREE_
#define _KD_TREE_

#include <iostream>
#include <cstdlib>

using namespace std;

#include <builtin_types.h>

struct kd_node {
  int   body_id;
  int   split;            // split_dimension
  float pos[3];
};

struct elem_type {
  int id;
  float pos[3];
};


void build_left_balanced_tree(int n,
			      kd_node *kd_tree,
			      float4  *bodies_pos);
void build_kd_hash(int n_bodies, kd_node *kd_tree, int kd_hash[]);

#endif // _KD_TREE_
