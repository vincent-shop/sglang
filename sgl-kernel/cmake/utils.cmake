# Adapt from: https://github.com/neuralmagic/vllm-flash-attention/blob/main/cmake/utils.cmake
#
# Clear all `-gencode` flags from `CMAKE_CUDA_FLAGS` and store them in
# `CUDA_ARCH_FLAGS`.
#
# Example:
#   CMAKE_CUDA_FLAGS="-Wall -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75"
#   clear_cuda_arches(CUDA_ARCH_FLAGS)
#   CUDA_ARCH_FLAGS="-gencode arch=compute_70,code=sm_70;-gencode arch=compute_75,code=sm_75"
#   CMAKE_CUDA_FLAGS="-Wall"
#
macro(clear_cuda_arches CUDA_ARCH_FLAGS)
    # Extract all `-gencode` flags from `CMAKE_CUDA_FLAGS`
    string(REGEX MATCHALL "-gencode arch=[^ ]+" CUDA_ARCH_FLAGS "${CMAKE_CUDA_FLAGS}")

    # Remove all `-gencode` flags from `CMAKE_CUDA_FLAGS` since they will be modified
    # and passed back via the `CUDA_ARCHITECTURES` property.
    string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endmacro()

function(parse_torch_cuda_arch_list OUTPUT_VAR RAW_LIST)
    if("${RAW_LIST}" STREQUAL "")
        set(${OUTPUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    string(REGEX REPLACE "[,;]" " " _normalized "${RAW_LIST}")
    string(REGEX REPLACE "[ \t]+" ";" _normalized "${_normalized}")
    string(REGEX REPLACE "^;|;$" "" _normalized "${_normalized}")

    set(_gencode_flags "")
    set(_invalid_entry FALSE)

    foreach(_entry ${_normalized})
        string(STRIP "${_entry}" _entry)
        if(_entry STREQUAL "")
            continue()
        endif()

        set(_add_ptx FALSE)
        if(_entry MATCHES "\\+ptx$" OR _entry MATCHES "\\+PTX$")
            set(_add_ptx TRUE)
            string(REGEX REPLACE "\\+ptx$|\\+PTX$" "" _entry "${_entry}")
        endif()

        string(TOLOWER "${_entry}" _entry_lower)
        if(NOT _entry_lower MATCHES "^[0-9]+(\\.[0-9]+)?[a-z]?$")
            set(_invalid_entry TRUE)
            continue()
        endif()

        string(REGEX MATCH "^([0-9]+)(\\.([0-9]+))?([a-z]?)$" _ "${_entry_lower}")
        set(_major "${CMAKE_MATCH_1}")
        set(_minor "${CMAKE_MATCH_3}")
        set(_suffix "${CMAKE_MATCH_4}")

        if(_minor STREQUAL "")
            set(_minor "0")
        endif()

        string(CONCAT _code "${_major}" "${_minor}" "${_suffix}")
        set(_compute "compute_${_code}")
        set(_sm "sm_${_code}")

        list(APPEND _gencode_flags "-gencode=arch=${_compute},code=${_sm}")
        if(_add_ptx)
            list(APPEND _gencode_flags "-gencode=arch=${_compute},code=${_compute}")
        endif()
    endforeach()

    if(_invalid_entry)
        message(WARNING "Ignoring unrecognized entries from TORCH_CUDA_ARCH_LIST='${RAW_LIST}'")
    endif()

    list(REMOVE_DUPLICATES _gencode_flags)
    set(${OUTPUT_VAR} "${_gencode_flags}" PARENT_SCOPE)
endfunction()
