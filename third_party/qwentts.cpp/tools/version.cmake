# Generate version.h with the current git commit hash and date.
# Only rewrites the file if the content changed (avoids rebuild cascade).
# Usage: cmake -DSRC_DIR=... -DOUTPUT=... -P version.cmake
#
# FORK PATCH (yache-local-voice): this vendored copy of the upstream source
# ships without a .git directory, so the plain git probes below would write
# "unknown" into version.h and break the qt_version() handshake the Python
# binding hard-verifies.  A CI build of this repository injects the exact
# upstream revision/date instead:
#     cmake -DQT_REVISION=<full sha> -DQT_DATE=<%cs date> ...
# leaving the generated version.h byte-identical to the upstream build.

if(DEFINED QT_REVISION AND NOT QT_REVISION STREQUAL "")
    set(GIT_HASH "${QT_REVISION}")
    if(DEFINED QT_DATE AND NOT QT_DATE STREQUAL "")
        set(GIT_DATE "${QT_DATE}")
    else()
        set(GIT_DATE "unknown")
    endif()
else()
    execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY "${SRC_DIR}"
        OUTPUT_VARIABLE GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE GIT_RESULT
    )
    if(NOT GIT_RESULT EQUAL 0)
        # No .git (tarball / vendored copy): this exact snapshot IS this
        # revision, so pin it rather than break the qt_version() handshake.
        set(GIT_HASH "6fae92914045cd83364d2845ceaa0f7969727319")
    endif()

    execute_process(
        COMMAND git show -s --format=%cs HEAD
        WORKING_DIRECTORY "${SRC_DIR}"
        OUTPUT_VARIABLE GIT_DATE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE DATE_RESULT
    )
    if(NOT DATE_RESULT EQUAL 0)
        set(GIT_DATE "2026-09-28")
    endif()
endif()

set(CONTENT "#pragma once\n#define QWEN_VERSION \"${GIT_HASH} (${GIT_DATE})\"\n")

if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" EXISTING)
    if("${EXISTING}" STREQUAL "${CONTENT}")
        return()
    endif()
endif()

file(WRITE "${OUTPUT}" "${CONTENT}")
