#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
 
void* say_hello(void* data)
{
    char *str;
    str = (char*)data;
    while(1)
    {
        printf("%s\n",str);
        usleep(1);
    }
}
 
void* open_file(void* data)
{
    char *str;
    str = (char*)data;
    printf("Opening File\n");
    FILE* f = fopen(str,"w");
    fclose(f);
    printf("Closing File\n");
    pthread_exit(NULL);
}
 
int main()
{
    pthread_t t1,t2,t3;
 
    pthread_create(&t1,NULL,open_file,"hello.txt");
    pthread_create(&t2,NULL,say_hello,"The CPU belongs to Thread 2 :D");
    pthread_create(&t3,NULL,say_hello,"The CPU belongs to Thread 3 :D");
    pthread_join(t1,NULL);
    return EXIT_SUCCESS;
}
