
#ifndef GLOBAL_IS_DEF
#define GLOBAL_IS_DEF

#include <stdbool.h>

extern unsigned display;
extern unsigned vsync;
extern unsigned refresh_rate;
extern int max_iter;
extern unsigned do_first_touch;
extern char *pngfile;
extern char *draw_param;

extern unsigned DIM;
extern unsigned GRAIN;

extern bool **change;
extern bool **next_change;

#endif
