#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <openssl/md5.h>
#define MaxPassLength 10

const char *string = "Hello MD5 Hash";

int main()
{
    unsigned char result[MD5_DIGEST_LENGTH];
    printf("%s\nMD5: ", string);
    MD5(string, strlen(string), result);
    for(int i = 0; i < MD5_DIGEST_LENGTH; i++)
    {
        printf("%02x", result[i]);
    }
    printf("\n\n");
    return EXIT_SUCCESS;
}