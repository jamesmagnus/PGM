//#ifdef cl_khr_fp64
//    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#elif defined(cl_amd_fp64)
//    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
//#else
//    #warning "Double precision floating point not supported by OpenCL implementation."
//#endif

#define RED 0xAA0000FF
#define YELLOW 0xFFFF00FF
#define GREEN 0x00AA00FF
#define BLACK 0x0

#define COLOR_TILES 0

// NE PAS MODIFIER
static int4 color_to_int4 (unsigned c)
{
    uchar4 ci = *(uchar4 *) &c;

    return convert_int4 (ci);
}

// NE PAS MODIFIER
static unsigned int4_to_color (int4 i)
{
    return (unsigned) convert_uchar4 (i);
}

static unsigned color_mean (unsigned c1,
                            unsigned c2)
{
    return int4_to_color ((color_to_int4 (c1) + color_to_int4 (c2)) / (int4)2);
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// scrollup
////////////////////////////////////////////////////////////////////////////////

__kernel void scrollup (__global unsigned *in,
                        __global unsigned *out)
{
    int y = get_global_id (1);
    int x = get_global_id (0);
    unsigned couleur;

    couleur = in [y * DIM + x];

    y = (y ? y - 1 : get_global_size (1) - 1);

    out [y * DIM + x] = couleur;
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// vie
////////////////////////////////////////////////////////////////////////////////



static unsigned compute_new_color(__global unsigned *in,
                                  int x,
                                  int y)
{
    unsigned couleur = 0;

    for (int i = y - 1; i <= y + 1; i++)
        for (int j = x - 1; j <= x + 1; j++)
            if (( i >= 0) && ( i < DIM) && ( j >= 0) && ( j < DIM) )
                couleur += (in[i * DIM + j] == YELLOW);

    if (in[y * DIM + x] == YELLOW) {
        if ((couleur == 3) || (couleur == 4))
            couleur = YELLOW;
        else
            couleur = BLACK;
    }
    else {
        if (couleur == 3)
            couleur = YELLOW;
        else
            couleur = BLACK;
    }
    return couleur;
}

__kernel void vie (__global unsigned *in,
                   __global unsigned *out,
                   __global char *change,
                   __global char *next_change)
{
    int x = get_global_id (0);
    int y = get_global_id (1);

    out [y * DIM + x] = compute_new_color(in, x, y);
}


__kernel void vie_opt (__global unsigned *in,
                       __global unsigned *out,
                       __global char *change,
                       __global char *next_change)
{
    int x = get_global_id (0);
    int y = get_global_id (1);

    int xglob = get_global_id (0);
    int yglob = get_global_id (1);

    int xloc = get_local_id (0);
    int yloc = get_local_id (1);

    int tx = (xglob - xloc) / TILEX;
    int ty = (yglob - yloc) / TILEY;

    __local char to_update;
    __local char changed;


    // Check if this tile has to be updated
    if (( xloc == 0) && ( yloc == 0) ) {
      to_update = 0;
      changed = 0;

      for (int x = tx - 1; x <= tx + 1; ++x)
          if (( x >= 0) && ( x < DIM / TILEX) )
              for (int y = ty - 1; y <= ty + 1; ++y)
                  if (( y >= 0) && ( y < DIM / TILEY)
                      && change[y * (DIM / TILEY) + x])
                      to_update = 1;
        
    }

// Be sure that every thread of the group wait
// for to_update being computed
    barrier(CLK_LOCAL_MEM_FENCE);


    if (to_update) {
        unsigned color = compute_new_color(in, xglob, yglob);
        if (color != in[yglob * DIM + xglob] 
  #if COLOR_TILES 
        && !(color == BLACK && in[yglob * DIM + xglob] == GREEN)
  #endif  
        ) {
          changed = 1;
        }
        out[yglob * DIM + xglob] = color;
    }
    else {
  #if COLOR_TILES
        out[yglob * DIM + xglob] = GREEN;
  #else
        out[yglob * DIM + xglob] = in[yglob * DIM + xglob];
  #endif
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (xloc == 0 && yloc == 0) 
      next_change[ty * (DIM/TILEY) + tx] = changed;

}

// NE PAS MODIFIER
static float4 color_scatter (unsigned c)
{
    uchar4 ci;

    ci.s0123 = (*((uchar4 *) &c)).s3210;
    return convert_float4 (ci) / (float4) 255;
}

// NE PAS MODIFIER: ce noyau est appelé lorsqu'une mise à jour de la
// texture de l'image affichée est requise
__kernel void update_texture (__global unsigned *cur,
                              __write_only image2d_t tex)
{
    int y = get_global_id (1);
    int x = get_global_id (0);
    int2 pos = (int2)(x, y);
    unsigned c = cur [y * DIM + x];

    write_imagef (tex, pos, color_scatter (c));
}
