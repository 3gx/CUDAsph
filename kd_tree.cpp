#include "kd_tree.h"
#include <stdio.h>

void q_sort(elem_type data[], int dim, int left, int right) {
  int l_hold = left;
  int  r_hold = right;
  elem_type pivot = data[left];

  while (left < right){
    while ((data[right].pos[dim] >= pivot.pos[dim]) && (left < right))
      right--;
    if (left != right) {
      data[left] = data[right];
      left++;
    }
    while ((data[left].pos[dim] <= pivot.pos[dim]) && (left < right))
      left++;
    if (left != right) {
      data[right] = data[left];
      right--;
    }
  }
  data[left] = pivot;
  int i_pivot = left;
  left = l_hold;
  right = r_hold;
  if (left < i_pivot)
    q_sort(data, dim, left, i_pivot-1);
  if (right > i_pivot)
    q_sort(data, dim, i_pivot+1, right);
}


void quickSort(elem_type data[], int dim, int array_size) {
  q_sort(data, dim, 0, array_size - 1);
}

int max_nodes, max_depth; 
void recursively_build_left_balanced_tree(kd_node *kd_tree, int n_node,
					  elem_type *data,
					  int depth, int n) {
  max_nodes = max(n_node, max_nodes);
  max_depth = max(max_depth, depth);

  int split = depth%3;

//   float r_min[3] = {data[0].pos[0], data[0].pos[1], data[0].pos[2]};
//   float r_max[3] = {data[0].pos[0], data[0].pos[1], data[0].pos[2]};
//   for (int i = 0; i < n; i++) {
//     for (int k = 0; k < 3; k++) {
//       r_min[k] = min(r_min[k], data[i].pos[k]);
//       r_max[k] = max(r_max[k], data[i].pos[k]);
//     }
//   }
//   float size[3];
//   for (int k = 0; k < 3; k++) 
//     size[k] = r_max[k] - r_min[k];
  
//   split = 0;
//   if (size[1] > size[0]) split = 1;
//   if (size[2] > size[1]) split = 2;

  quickSort(data, split, n);
  
  int m = 1;
  while (m <= n) m = (m << (1));
  m = (m >> (1));
  
  int r = n - (m - 1);
  int lt, rt;
  if (r <= m/2) {
      lt = (m-2)/2 + r;
      rt = (m-2)/2;
  } else {
    lt = (m-2)/2 + m/2;
    rt = (m-2)/2 - m/2 + r;
  }
  
  int median = lt;
  elem_type element = data[median];
  int n_left  = median;
  int n_right = n - median - 1;
  
  kd_node &node = kd_tree[n_node];  
  node.split  = split;
  node.body_id = element.id;
    
  for (int k = 0; k < 3; k++)
    node.pos[k] = element.pos[k];
  
  if (n_left > 0) 
    recursively_build_left_balanced_tree(kd_tree, 2*n_node, &data[0], depth + 1, n_left);
  if (n_right > 0) 
    recursively_build_left_balanced_tree(kd_tree, 2*n_node + 1, &data[median+1], depth+1, n_right);
}
  
void build_left_balanced_tree(int n,
			      kd_node *kd_tree,
			      float4  *bodies_pos) {
  elem_type *data = (elem_type*)malloc(n*sizeof(elem_type));
  
  for (int i = 0; i < n; i++) {
    data[i].id     = i;
    data[i].pos[0] = bodies_pos[i].x;
    data[i].pos[1] = bodies_pos[i].y;
    data[i].pos[2] = bodies_pos[i].z;
  } 
  
  max_depth = max_nodes = 0;
  int depth = 0;
  recursively_build_left_balanced_tree(kd_tree, 1, data, depth, n);
  fprintf(stderr, "<tree info: max_depth= %d  max_nodes= %d> ... ", max_depth, max_nodes);
  free(data);
}


void build_kd_hash(int n_bodies, kd_node *kd_tree, int kd_hash[]) {
  int i_begin = 1;
  int i_map = 1;
  for (int i = 0; i < n_bodies; i++) {
    if (i_map > n_bodies) {
      if (i_begin%2 == 0) i_begin += 1;
      else	          i_begin = (i_begin-1)*2; 
      i_map = i_begin;
    }
    kd_hash[i] = kd_tree[i_map++].body_id;
  }
}
