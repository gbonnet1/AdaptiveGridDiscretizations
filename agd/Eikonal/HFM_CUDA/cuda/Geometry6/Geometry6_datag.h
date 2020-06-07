#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0


// These ones are the changes of variables toward the reference forms in the neighbors
const small chg1_1[6][6] = 
 {{0,0,1,0,0,0},
 {0,0,0,1,0,0},
 {1,0,0,0,0,0},
 {0,1,0,0,0,0},
 {0,0,0,0,1,0},
 {0,0,0,0,0,1}} ;
const small chg1_3[6][6] = 
 {{0,0,1,0,0,0},
 {0,0,0,1,0,0},
 {0,0,0,0,1,0},
 {1,0,0,0,0,0},
 {0,1,0,0,0,0},
 {0,0,0,0,0,1}} ;
const small chg1_6[6][6] = 
 {{ 1, 1, 0, 0, 1, 1},
 { 1, 0, 0, 0, 0, 0},
 { 0, 1, 0, 0, 0, 0},
 { 0, 0, 1, 0, 0, 0},
 { 0, 0, 0, 1, 0, 0},
 {-1,-1,-1, 0, 0,-1}} ;
const small chg2_0[6][6] = 
 {{ 0, 0, 1, 1, 1, 1},
 { 0, 0, 0, 0, 1, 0},
 { 0, 0, 0, 0, 0, 1},
 { 1, 0, 0, 0, 0, 0},
 { 0, 1, 0, 0, 0, 0},
 {-1, 0, 0,-1,-1,-1}} ;
const small chg2_1[6][6] = 
 {{ 1, 0, 0, 0, 0, 0},
 { 1,-1, 0, 0, 0, 0},
 { 0, 0, 1, 0, 0, 0},
 { 0, 0, 0, 1, 0, 0},
 { 0, 0, 0, 0, 1, 0},
 {-1, 1, 0, 0, 0, 1}} ;
const small chg2_2[6][6] = 
 {{ 1, 0, 0, 0, 0, 0},
 { 2, 1, 0, 1, 1, 1},
 { 1, 0, 1, 1, 0, 1},
 {-1, 0, 0,-1, 0, 0},
 { 0, 0, 0, 0, 1, 0},
 {-2, 0, 0, 0,-1,-1}} ;
const small chg2_4[6][6] = 
 {{0,0,0,1,0,0},
 {0,0,0,0,1,0},
 {1,0,0,0,0,0},
 {0,1,0,0,0,0},
 {0,0,1,0,0,0},
 {0,0,0,0,0,1}} ;
const small chg2_5[6][6] = 
 {{ 0, 1, 0, 0, 0, 0},
 { 0, 0, 1, 0, 0, 0},
 { 0, 0, 0, 1, 0, 0},
 { 0, 0, 0, 0, 1, 0},
 { 1, 0, 0, 1, 0, 1},
 { 0,-1,-1,-1, 0,-1}} ;
const small chg2_6[6][6] = 
 {{ 0, 0, 0, 1, 0, 0},
 { 1, 0, 0, 1, 0, 1},
 { 0, 0, 0, 0, 1, 0},
 { 0,-1,-1,-1, 0,-1},
 { 0, 1, 0, 0, 0, 0},
 { 0, 0, 1, 0, 0, 0}} ;
const small chg2_7[6][6] = 
 {{ 1, 0, 0, 0, 0, 0},
 { 0, 1, 0, 0, 0, 0},
 {-1,-1,-1,-1,-1,-1},
 { 0, 0, 0, 1, 0, 0},
 { 0, 0, 0, 0, 1, 0},
 { 0, 0, 0, 0, 0, 1}} ;
const small chg2_8[6][6] = 
 {{ 1, 0, 0, 0, 0, 0},
 { 0, 1, 0, 0, 0, 0},
 {-1,-1,-1,-1,-1,-1},
 { 0, 0, 0, 1, 0, 0},
 { 0, 0, 0, 0, 1, 0},
 { 0, 0, 0, 0, 0, 1}} ;
const small chg2_9[6][6] = 
 {{0,0,0,1,0,0},
 {0,0,0,0,1,0},
 {1,0,0,0,0,0},
 {0,1,0,0,0,0},
 {0,0,1,0,0,0},
 {0,0,0,0,0,1}} ;
const small chg3_0[6][6] = 
 {{ 1, 0, 1, 0, 0, 0},
 { 1, 0, 0, 1, 0, 0},
 {-1, 0, 0, 0, 0, 0},
 { 0, 0, 0, 0, 1, 0},
 { 0, 0, 0, 0, 0, 1},
 { 0, 1, 0, 0, 0, 0}} ;
const small chg3_1[6][6] = 
 {{ 0, 1, 0, 1, 1, 1},
 { 0, 0, 0, 0, 1, 0},
 { 0, 0, 0, 0, 0, 1},
 { 1, 0, 0, 0, 0, 0},
 { 0, 0, 1, 0, 0, 0},
 {-1, 0, 0,-1,-1,-1}} ;
const small chg3_2[6][6] = 
 {{0,0,0,0,1,0},
 {0,0,0,0,0,1},
 {1,0,0,0,0,0},
 {0,1,0,0,0,0},
 {0,0,1,0,0,0},
 {0,0,0,1,0,0}} ;
const small chg4_1[6][6] = 
 {{ 1, 0, 0, 0, 0, 0},
 { 0, 1, 0, 0, 0, 0},
 {-1,-1,-1,-1,-1,-1},
 { 0, 0, 0, 1, 0, 0},
 { 0, 0, 0, 0, 1, 0},
 { 0, 0, 0, 0, 0, 1}} ;
const small chg4_2[6][6] = 
 {{ 0,-1,-1,-1, 0, 0},
 {-1, 0,-1,-1, 0, 0},
 { 0, 0, 1, 0, 0, 0},
 { 0, 0, 0, 1, 0, 0},
 { 1, 1, 1, 1, 1, 1},
 { 0, 0, 0, 0, 0,-1}} ;
const small chg5_1[6][6] = 
 {{ 1, 0, 0, 0, 0, 0},
 { 0, 1, 0, 0, 0, 0},
 {-1,-1,-1,-1,-1,-1},
 { 0, 0, 0, 1, 0, 0},
 { 0, 0, 0, 0, 1, 0},
 { 0, 0, 0, 0, 0, 1}} ;
const small chg5_2[6][6] = 
 {{0,0,1,0,0,0},
 {0,0,0,1,0,0},
 {0,0,0,0,1,0},
 {1,0,0,0,0,0},
 {0,1,0,0,0,0},
 {0,0,0,0,0,1}} ;
const small chg5_3[6][6] = 
 {{ 0,-1, 0, 0,-1, 0},
 {-1, 0, 0, 0,-1, 0},
 { 0, 0, 0, 0, 1, 0},
 { 1, 1, 1, 1, 1, 1},
 { 0, 0,-1, 0, 0, 0},
 { 0, 0, 0,-1, 0, 0}} ;
const small chg5_4[6][6] = 
 {{ 1, 0, 0, 0, 0, 1},
 { 0, 1, 0, 0, 0, 1},
 { 0, 0, 1, 0, 0, 0},
 { 0, 0, 0, 1, 0, 0},
 { 0, 0, 0, 0, 1, 0},
 { 0, 0, 0, 0, 0,-1}} ;


typedef const small (*chgi_jT)[6]; // small[6][6]

const int neigh0_base_v[1] = {1} ;
const chgi_jT neigh0_base_c[1] = {nullptr} ;
const int neigh1_base_v[8] = {0, 1, 2, 2, 3, 5, 2, 4} ;
const chgi_jT neigh1_base_c[8] = {nullptr, chg1_1, nullptr, chg1_3, nullptr, nullptr, chg1_6, nullptr} ;
const int neigh2_base_v[11] = {2, 2, 2, 1, 1, 1, 3, 4, 5, 5, 6} ;
const chgi_jT neigh2_base_c[11] = {chg2_0, chg2_1, chg2_2, nullptr, chg2_4, chg2_5, chg2_6, chg2_7, chg2_8, chg2_9, nullptr} ;
const int neigh3_base_v[3] = {5, 2, 1} ;
const chgi_jT neigh3_base_c[3] = {chg3_0, chg3_1, chg3_2} ;
const int neigh4_base_v[3] = {1, 2, 5} ;
const chgi_jT neigh4_base_c[3] = {nullptr, chg4_1, chg4_2} ;
const int neigh5_base_v[5] = {1, 2, 2, 3, 4} ;
const chgi_jT neigh5_base_c[5] = {nullptr, chg5_1, chg5_2, chg5_3, chg5_4} ;
const int neigh6_base_v[1] = {2} ;
const chgi_jT neigh6_base_c[1] = {nullptr} ;
