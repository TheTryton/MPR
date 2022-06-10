
#include <stdio.h>


 int scr_read_pgm(const char* name, unsigned char* image, int irows, int icols );
 void scr_write_pgm(const char* name, unsigned char* image, int rows, int cols, const char* comment );
 int scr_read_ppm(const char* name, unsigned char* image, int irows, int icols );
 void scr_write_ppm(const char* name, unsigned char* image, int rows, int cols, const char* comment );
 void get_PgmPpmParams(const char * , int *, int *);
 void getout_comment(FILE * );
