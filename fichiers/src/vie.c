#include "global.h"
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"
#include "scheduler.h"
#include "constants.h"

#include <stdbool.h>

#define COLOR_TILES 1

static void compute_new_state (int y,
                               int x)
{
    unsigned n = 0;

    if ((x > 0) && (x < DIM - 1) && (y > 0) && (y < DIM - 1)) {
        for (int i = y - 1; i <= y + 1; i++)
            for (int j = x - 1; j <= x + 1; j++)
                n += (cur_img (i, j) == YELLOW);

        if (cur_img (y, x) == YELLOW) {
            if ((n == 3) || (n == 4))
                n = YELLOW;
            else
                n = BLACK;
        }
        else {
            if (n == 3)
                n = YELLOW;
            else
                n = BLACK;
        }

        next_img (y, x) = n;
    }
}

///////////////////////////////////////////////////////////////////////////////

// Version séquentielle simple
unsigned vie_compute_seq (unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++) {
        for (int i = 0; i < DIM; i++)
            for (int j = 0; j < DIM; j++)
                compute_new_state (i, j);

        swap_images ();
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////

// Version séquentielle avec tuiles
unsigned vie_compute_seq_tile (unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++) {
        for (int tx = 0; tx < DIM / TILEX; ++tx)
            for (int ty = 0; ty < DIM / TILEY; ++ty)
                for (int i = tx * TILEX; i < (tx + 1) * TILEX; i++)
                    for (int j = ty * TILEY; j < (ty + 1) * TILEY; j++)
                        compute_new_state (i, j);

        swap_images ();
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////

bool tile_need_update(int tx,
                      int ty)
{
    for (int x = tx - 1; x <= tx + 1; ++x)
        for (int y = ty - 1; y <= ty + 1; ++y)
            if ((x >= 0) && (y >= 0) && (x < (DIM / TILEX)) &&
                ( y < (DIM / TILEY)) && change[x][y])
                return true;
    return false;
}

void color_tile(int tx,
                int ty,
                unsigned color)
{
    for (int i = tx * TILEX; i < (tx + 1) * TILEX; i++)
        for (int j = ty * TILEY; j < (ty + 1) * TILEY; j++)
            if (next_img(i, j) != YELLOW)
                next_img(i, j) = color;
}

void swap_changes()
{
    bool **tmp = change;

    change = next_change;
    next_change = tmp;
}

void show_tile_color_debug()
{
    for (int tx = 0; tx < DIM / TILEX; ++tx)
        for (int ty = 0; ty < DIM / TILEY; ++ty)
            color_tile(tx, ty, BLACK);

    for (int tx = 0; tx < DIM / TILEX; ++tx) {
        for (int ty = 0; ty < DIM / TILEY; ++ty)
            if (next_change[tx][ty]) {
                color_tile(tx, ty, RED);

                for (int x = tx - 1; x <= tx + 1; ++x)
                    for (int y = ty - 1; y <= ty + 1; ++y)
                        if ((x >= 0) && (y >= 0) && (x < (DIM / TILEX)) &&
                            ( y < (DIM / TILEY)) &&
                            (( x != tx) || ( y != ty) ) &&
                            !next_change[x][y])
                            color_tile(x, y, GREEN);
            }
    }
}

void reset_next_change()
{
    for (int tx = 0; tx < DIM / TILEX; ++tx)
        for (int ty = 0; ty < DIM / TILEY; ++ty)
            next_change[tx][ty] = false;
}

void update_tile(int tx,
                 int ty)
{
    for (int i = tx * TILEX; i < (tx + 1) * TILEX; i++)
        for (int j = ty * TILEY; j < (ty + 1) * TILEY; j++) {
            int prec = cur_img (i, j);
            compute_new_state (i, j);
            if (!next_change[tx][ty] &&
                ((( prec == BLACK) &&
                  ( next_img(i,
                             j) == YELLOW) ) ||
                 (( prec == YELLOW) &&
                  ( next_img(i, j) == BLACK) )))
                next_change[tx][ty] = true;
        }
}

// Version séquentielle optimisée avec tuiles
unsigned vie_compute_seq_tile_opt (unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++) {
        reset_next_change();
        for (int tx = 0; tx < DIM / TILEX; ++tx)
            for (int ty = 0; ty < DIM / TILEY; ++ty)
                if (tile_need_update(tx, ty))
                    update_tile(tx, ty);

#if COLOR_TILES
        show_tile_color_debug();
#endif

        swap_changes(); //swap changes and next_changes boolean matrixs
        swap_images();
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////

// Version OMP simple
unsigned vie_compute_omp(unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < DIM; i++)
            for (int j = 0; j < DIM; j++)
                compute_new_state (i, j);

        swap_images ();
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////

// Version OMP avec tuiles
unsigned vie_compute_omp_tile (unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int tx = 0; tx < DIM / TILEX; ++tx)
            for (int ty = 0; ty < DIM / TILEY; ++ty)
                for (int i = tx * TILEX; i < (tx + 1) * TILEX; i++)
                    for (int j = ty * TILEY; j < (ty + 1) * TILEY; j++)
                        compute_new_state (i, j);

        swap_images ();
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////

// Version OMP optimisée avec tuiles
unsigned vie_compute_omp_tile_opt (unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++) {
        reset_next_change();
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int tx = 0; tx < DIM / TILEX; ++tx)
            for (int ty = 0; ty < DIM / TILEY; ++ty)
                if (tile_need_update(tx, ty))
                    update_tile(tx, ty);

#if COLOR_TILES
        show_tile_color_debug();
#endif

        swap_changes(); //swap changes and next_changes boolean matrixs
        swap_images();
    }

    return 0;
}

///////////////////////////// Configuration initiale

void draw_stable (void);
void draw_guns (void);
void draw_random (void);

void vie_draw (char *param)
{
    char func_name [1024];

    void (*f)(void) = NULL;

    sprintf (func_name, "draw_%s", param);
    f = dlsym (DLSYM_FLAG, func_name);

    if (f == NULL) {
        printf ("Cannot resolve draw function: %s\n", func_name);
        f = draw_guns;
    }

    f ();
}

static unsigned couleur = 0xFFFF00FF; // Yellow

static void gun (int x,
                 int y,
                 int version)
{
    bool glider_gun [11][38] =
    {
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 },
        { 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    };

    if (version == 0)
        for (int i = 0; i < 11; i++)
            for (int j = 0; j < 38; j++)
                if (glider_gun [i][j])
                    cur_img (i + x, j + y) = couleur;

    if (version == 1)
        for (int i = 0; i < 11; i++)
            for (int j = 0; j < 38; j++)
                if (glider_gun [i][j])
                    cur_img (x - i, j + y) = couleur;

    if (version == 2)
        for (int i = 0; i < 11; i++)
            for (int j = 0; j < 38; j++)
                if (glider_gun [i][j])
                    cur_img (x - i, y - j) = couleur;

    if (version == 3)
        for (int i = 0; i < 11; i++)
            for (int j = 0; j < 38; j++)
                if (glider_gun [i][j])
                    cur_img (i + x, y - j) = couleur;
}

void draw_stable (void)
{
    for (int i = 1; i < DIM - 2; i += 4)
        for (int j = 1; j < DIM - 2; j += 4)
            cur_img (i, j) = cur_img (i, (j + 1)) = cur_img ((i + 1), j) = cur_img ((i + 1), (j + 1)) = couleur;
}

void draw_guns (void)
{
    memset(&cur_img (0, 0), 0, DIM * DIM * sizeof(cur_img (0, 0)));

    gun (0, 0, 0);
    gun (0, DIM - 1, 3);
    gun (DIM - 1, DIM - 1, 2);
    gun (DIM - 1, 0, 1);
}

void draw_random (void)
{
    for (int i = 1; i < DIM - 1; i++)
        for (int j = 1; j < DIM - 1; j++)
            cur_img (i, j) = (random() & 01) ? couleur : 0;
}

// Une tête de clown apparaît à l'itération 110
void draw_clown (void)
{
    int i = DIM / 2, j = i;

    cur_img (i, j - 1) = couleur;
    cur_img (i, j) = couleur;
    cur_img (i, j + 1) = couleur;

    cur_img (i + 1, j - 1) = couleur;
    cur_img (i + 1, j + 1) = couleur;

    cur_img (i + 2, j - 1) = couleur;
    cur_img (i + 2, j + 1) = couleur;
}
