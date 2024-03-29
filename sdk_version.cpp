/**********************************************************************
*
* <COPYRIGHT_TAG>
*
**********************************************************************/

/*
 * @file   sdk_version.cpp
 * @brief  Report versions of all SDK modules
*/

#include <stdint.h>
#include <string.h>
#include "sdk_version.h"

int16_t
bblib_sdk_version(char **buffer, const char **version, int buffer_size)
{
    /* Check that the version string is set and that the buffer is
       sufficiently large */
    if (buffer_size < 1)
        return -1;

    if (!*version || (buffer_size <= strlen(*version))) {
        strncpy(*buffer, "", buffer_size-1);
        return -1;
    }

    strncpy(*buffer, *version, buffer_size-1);
    return 0;
}


struct bblib_common_init
{
    bblib_common_init()
    {
        bblib_print_common_version();
    }
};

bblib_common_init do_constructor_common;




int16_t
bblib_common_version(char *version, int buffer_size)
{
    /* The version string will be updated before the build process starts by the
     *       jobs building the library and/or preparing the release packages.
     *       Do not edit the version string manually */
    const char *msg = "FlexRAN SDK bblib_common version #DIRTY#";

    return(bblib_sdk_version(&version, &msg, buffer_size));
}

void
bblib_print_common_version()
{
    static bool was_executed = false;
    if(!was_executed) {
        was_executed = true;
        char version[BBLIB_SDK_VERSION_STRING_MAX_LEN] = { };
        bblib_common_version(version, sizeof(version));
        printf("%s\n", version);
    }
}
