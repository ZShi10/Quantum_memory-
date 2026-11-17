#include <archive.h>
#include <archive_entry.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
int main(int argc,char**argv){if(argc!=3)return 1;mkdir(argv[2],0775);if(chdir(argv[2]))return 2;struct archive*a=archive_read_new();archive_read_support_filter_all(a);archive_read_support_format_all(a);if(archive_read_open_filename(a,argv[1],10240)) {fprintf(stderr,"%s\n",archive_error_string(a));return 3;}struct archive_entry*e;int n=0;while(archive_read_next_header(a,&e)==ARCHIVE_OK){const char*p=archive_entry_pathname(e);if(p[0]=='/'||strstr(p,"../"))return 4;int r=archive_read_extract(a,e,ARCHIVE_EXTRACT_SECURE_SYMLINKS|ARCHIVE_EXTRACT_SECURE_NODOTDOT);if(r!=ARCHIVE_OK){fprintf(stderr,"%s\n",archive_error_string(a));return 5;}n++;}archive_read_free(a);printf("Extracted %d entries\n",n);return 0;}
